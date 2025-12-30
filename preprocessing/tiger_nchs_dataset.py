#!/usr/bin/env python3
#
# tiger_nchs_dataset.py  Andrew Belles  Dec 24, 2025
#
# Processes road data on a per county basis from TIGER dataset.
# Computes 7 Structural Features (Density, Topology, Granularity, Shape).
# Parallelized for performance.
#

import argparse, os, fiona, time, math
import geopandas as gpd
import pandas as pd
import numpy as np 
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
from scipy.io import savemat 
from multiprocessing import Pool, cpu_count
from support.helpers import project_path

# for graph statistics  
import heapq
from collections import deque 

# Global variables for workers
WORKER_GDF = None
WORKER_SINDEX = None
WORKER_TRANSFORMER = None
WORKER_MTFCC_HWY = None
WORKER_LABEL_MAP = None

def _bearing_deg(p0, p1): 
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if dx == 0 and dy == 0: 
        return 0.0 
    ang = np.degrees(np.atan2(dy, dx))
    return (ang + 180.0) % 180.0


def init_worker(county_shp, src_crs, proj_crs, label_map):
    global WORKER_GDF, WORKER_SINDEX, WORKER_TRANSFORMER
    global WORKER_MTFCC_HWY, WORKER_LABEL_MAP
    
    # 1. Load Counties
    gdf = gpd.read_file(county_shp).to_crs(src_crs)
    gdf["FIPS"] = gdf["GEOID"]
    
    # Filter by labels (optimization)
    gdf = gdf[gdf["FIPS"].isin(label_map.keys())]
    
    WORKER_GDF = gdf
    WORKER_SINDEX = gdf.sindex
    WORKER_LABEL_MAP = label_map
    WORKER_TRANSFORMER = Transformer.from_crs(src_crs, proj_crs, always_xy=True).transform
    WORKER_MTFCC_HWY = {'S1100', 'S1200'}

def process_batch(features_batch):
    """
    Computes local stats including Euclidean distance for Circuity.
    """
    local_stats = {
        "matched_fips": {
            "len_hwy": 0.0, 
            "cnt_hwy": 0, 
            "len_local": 0.0, 
            "cnt_local": 0, 
            "nodes": {},
            "segments": []
        }
    }
    
    for props, geom_obj in features_batch:
        mtfcc = props.get('MTFCC', '')
        
        # 1. Spatial Lookup
        possible_idxs = list(WORKER_SINDEX.query(geom_obj.centroid, predicate='intersects'))
        if not possible_idxs: continue
            
        matched_fips = WORKER_GDF.iloc[possible_idxs[0]]["FIPS"]
        
        if matched_fips not in local_stats:
            local_stats[matched_fips] = {
                "len_hwy": 0.0, "cnt_hwy": 0,
                "len_local": 0.0, "cnt_local": 0,
                "euclid_local": 0.0, # Track straight-line dist for circuity
                "nodes": {},
                "segments": []
            }
        
        stats = local_stats[matched_fips]
        stats.setdefault("euclid_local", 0.0)
        stats.setdefault("segments", [])
        stats.setdefault("nodes", {})
        
        # 2. Project & Measure
        geom_proj = transform(WORKER_TRANSFORMER, geom_obj)
        length_m = geom_proj.length

        lines = []
        if geom_proj.geom_type == "LineString": 
            lines = [geom_proj]
        elif geom_proj.geom_type == "MultiLineString": 
            lines = list(geom_proj.geoms)

        for line in lines: 
            coords = list(line.coords)
            if len(coords) < 2: 
                continue 
            p0, p1 = coords[0], coords[-1]
            n0 = (round(p0[0], -1), round(p0[1], -1))
            n1 = (round(p1[0], -1), round(p1[1], -1))
            bearing = _bearing_deg(p0, p1)
            stats["segments"].append((n0, n1, float(line.length), bearing))

        
        if mtfcc in WORKER_MTFCC_HWY:
            stats["len_hwy"] += length_m
            stats["cnt_hwy"] += 1
        else:
            stats["len_local"] += length_m
            stats["cnt_local"] += 1
            
            # Compute Euclidean Distance (Start-to-End) for Circuity
            # If loop, dist is 0 (Circuity is infinite, handled by sum aggregation)
            if geom_proj.geom_type == "LineString":
                c = geom_proj.coords
                if len(c) >= 2:
                    dx = c[-1][0] - c[0][0]
                    dy = c[-1][1] - c[0][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    stats["euclid_local"] += dist
            elif geom_proj.geom_type == "MultiLineString":
                # For MultiLine, sum euclidean of parts
                for line in geom_proj.geoms:
                    c = line.coords
                    if len(c) >= 2:
                        dx = c[-1][0] - c[0][0]
                        dy = c[-1][1] - c[0][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        stats["euclid_local"] += dist
            
        # 3. Topology (Nodes)
        pts = []
        if geom_proj.geom_type == "LineString":
            c = geom_proj.coords
            if c: pts = [c[0], c[-1]]
        elif geom_proj.geom_type == "MultiLineString":
            for line in geom_proj.geoms:
                c = line.coords
                if c:
                    pts.append(c[0])
                    pts.append(c[-1])
        
        for pt in pts:
            # Snap to 10m grid
            key = (round(pt[0], -1), round(pt[1], -1))
            stats["nodes"][key] = stats["nodes"].get(key, 0) + 1
            
    return local_stats

class TigerNHCS:
    
    SRC_CRS = "EPSG:4269"
    PROJ_CRS = "EPSG:5070"
    BATCH_SIZE = 20000 
    
    def __init__(self, gdb_path, county_shp, out_path, labels_path=None, state_filter=None):
        self.gdb_path = gdb_path
        self.county_shp = county_shp
        self.out_path = out_path
        self.state_filter = state_filter # Not used in production run but good for debug
        
        if labels_path is None:
             labels_path = project_path("data", "nchs", "nchs_classification.csv")
        self.label_map = self._load_labels(labels_path)

    def _load_labels(self, path):
        labels = {}
        if not os.path.exists(path):
            print(f"[WARN] Labels file not found at {path}")
            return {}
        df = pd.read_csv(path, dtype=str)
        for _, row in df.iterrows():
            fips = row.get('FIPS', '').strip().zfill(5)
            code = row.get('class_code', '').strip()
            if fips and code and code.isdigit():
                labels[fips] = int(code) - 1 
        return labels

    def _find_edges_layer(self):
        layers = fiona.listlayers(self.gdb_path)
        for candidate in ["Edges", "Road", "AllRoads", "Line"]:
            for l in layers:
                if candidate.lower() in l.lower(): return l
        return layers[0]

    def run(self):
        start_time = time.time()
        n_cores = max(1, cpu_count() - 1) 
        print(f"[INFO] Initializing Pool with {n_cores} workers...")
        
        pool = Pool(
            processes=n_cores,
            initializer=init_worker,
            initargs=(self.county_shp, self.SRC_CRS, self.PROJ_CRS, self.label_map)
        )

        target_layer = self._find_edges_layer()
        print(f"[INFO] Streaming from GDB Layer: {target_layer}")
        
        batch = []
        jobs = []
        DRIVING = {'S1100', 'S1200', 'S1400'}
        
        with fiona.open(self.gdb_path, layer=target_layer) as src:
            for feature in src:
                if feature['properties'].get('MTFCC') not in DRIVING:
                    continue
                try:
                    geom = shape(feature['geometry'])
                    if not geom.is_empty:
                        batch.append((feature['properties'], geom))
                except: continue

                if len(batch) >= self.BATCH_SIZE:
                    jobs.append(pool.apply_async(process_batch, (batch,)))
                    batch = []
                    if len(jobs) % 10 == 0:
                        print(f"> Submitted {len(jobs) * self.BATCH_SIZE} roads...", end="\r")

            if batch: jobs.append(pool.apply_async(process_batch, (batch,)))

        print(f"\n[INFO] All tasks submitted. Waiting for results...")
        pool.close()
        pool.join()

        print("[INFO] Aggregating worker results...")
        final_stats = {} 
        
        for job in jobs:
            batch_result = job.get() 
            for fips, data in batch_result.items():
                if fips not in final_stats:
                    final_stats[fips] = {
                        "len_hwy": 0.0, "cnt_hwy": 0,
                        "len_local": 0.0, "cnt_local": 0,
                        "euclid_local": 0.0,
                        "nodes": {},
                        "segments": []
                    }
                
                t = final_stats[fips]
                t["len_hwy"] += data["len_hwy"]
                t["cnt_hwy"] += data["cnt_hwy"]
                t["len_local"] += data["len_local"]
                t["cnt_local"] += data["cnt_local"]
                t["euclid_local"] += data["euclid_local"]
                t["segments"].extend(data["segments"])
                
                for k, v in data["nodes"].items():
                    t["nodes"][k] = t["nodes"].get(k, 0) + v

        self._save(final_stats)
        print(f"[DONE] Total time: {time.time() - start_time:.2f}s")

    def _save(self, stats):
        gdf = gpd.read_file(self.county_shp)
        gdf["FIPS"] = gdf["GEOID"]
        
        rows = []
        for fips, data in stats.items():
            if fips not in self.label_map: continue
            
            try:
                row = gdf.loc[gdf["FIPS"]==fips]
                if row.empty: continue
                aland = row["ALAND"].values[0]
                area_km2 = float(aland) / 1e6
            except: continue
            
            if area_km2 < 0.1: area_km2 = 0.1
            
            # --- FEATURE 1-2: Densities ---
            dens_hwy = (data["len_hwy"]/1000)/area_km2
            dens_local = (data["len_local"]/1000)/area_km2
            
            # --- FEATURE 3-4: Avg Lengths ---
            cnt_local = data.get("cnt_local", 0)
            avg_len_local = (data["len_local"] / cnt_local) if cnt_local > 0 else 0.0
            
            cnt_hwy = data.get("cnt_hwy", 0)
            avg_len_hwy = (data["len_hwy"] / cnt_hwy) if cnt_hwy > 0 else 0.0
            
            # --- FEATURE 5: 4-Way Ratio (Grid) ---
            # --- FEATURE 6: Dead-End Density (Suburban) ---
            deg1, deg3, deg4 = 0, 0, 0
            for _, count in data["nodes"].items():
                if count == 1: deg1 += 1
                elif count == 3: deg3 += 1
                elif count >= 4: deg4 += 1
            
            total_int = deg3 + deg4

            ratio_4way = deg4 / total_int if total_int > 0 else 0.0 
            
            ratio_3way = deg3 / total_int if total_int > 0 else 0.0 
            
            # Dead End Density (Dead Ends / km2)
            dens_deadend = deg1 / area_km2
            
            # Circuity (Shape) ---
            # Ratio of Actual Length / Straight Line Length
            # 1.0 = Straight, >1.2 = Winding
            sum_euclid = data.get("euclid_local", 0)
            if sum_euclid > 0:
                circuity = data["len_local"] / sum_euclid
            else:
                circuity = 1.0 # Default to straight if no data
            
            # Meshedness Coefficient 
            V = len(data["nodes"])
            E = cnt_local 

            if V > 5: 
                meshedness = (E - V + 1) / (2 * V - 5)
            else: 
                meshedness = 0.0

            # custom features 
            bet_mean, bet_max, straightness, orient_entropy, integration = (
                self._compute_topology_metrics(data.get("segments", []))
            )

            rows.append({
                "FIPS": fips,
                "label": self.label_map[fips],
                "tiger_density_hwy": dens_hwy,
                "tiger_density_local": dens_local,
                "tiger_ratio_4way": ratio_4way, 
                "tiger_ratio_3way": ratio_3way, 
                "tiger_meshedness": meshedness, 
                "tiger_avg_len_local": avg_len_local,
                "tiger_avg_len_hwy": avg_len_hwy,
                "tiger_density_deadend": dens_deadend,
                "tiger_circuity_local": circuity, 
                "tiger_betweenness_mean": bet_mean, 
                "tiger_betweenness_max": bet_max, 
                "tiger_straightness_mean": straightness,
                "tiger_orientation_entropy": orient_entropy,
                "tiger_integration_r3": integration
            })
            
        df = pd.DataFrame(rows)
        if not df.empty:
            feature_cols = [
                "tiger_density_hwy", 
                "tiger_density_local", 
                "tiger_ratio_4way", 
                "tiger_ratio_3way", 
                "tiger_avg_len_local", 
                "tiger_avg_len_hwy",
                "tiger_density_deadend", 
                "tiger_circuity_local", 
                "tiger_meshedness",
                "tiger_betweenness_mean", 
                "tiger_betweenness_max", 
                "tiger_straightness_mean",
                "tiger_orientation_entropy",
                "tiger_integration_r3"
            ]
            mat = {
                "features": df[feature_cols].to_numpy(dtype=np.float64),
                "labels": df["label"].to_numpy(dtype=np.int64).reshape(-1, 1), 
                "feature_names": np.array(feature_cols, dtype="U"), 
                "fips_codes": df["FIPS"].to_numpy(dtype="U5"), 
                "n_counties": df.shape[0]
            }
            savemat(self.out_path, mat)
            print(f"Saved {len(df)} rows to {self.out_path}")
        else:
            print("No data produced.")

    # Custom Statistics

    def _compute_topology_metrics(self, segments): 
        if not segments: 
            return 0.0, 0.0, 0.0, 0.0, 0.0 
        rng = np.random.default_rng(0)
        dual_adj = self._build_dual_adjacency(segments)
        bc       = self._approx_betweenness(dual_adj, rng)
        bet_mean = float(np.mean(bc)) if bc.size else 0.0 
        bet_max  = float(np.max(bc)) if bc.size else 0.0 
        straightness   = self._straightness_mean(segments, rng)
        orient_entropy = self._orientation_entropy(segments)
        integration    = self._integration_mean(dual_adj, radius=3)
        return bet_mean, bet_max, straightness, orient_entropy, integration 

    @staticmethod 
    def _build_primal_graph(segments): 
        '''
        Builds graph w/ intersections as nodes and streets as edges between them 
        '''
        node_index = {}
        coords = []
        adj_w  = []

        def get_idx(node): 
            idx = node_index.get(node)
            if idx is None: 
                idx = len(coords)
                node_index[node] = idx 
                coords.append(node)
                adj_w.append([])
            return idx 

        for a, b, length, _ in segments: 
            ia = get_idx(a)
            ib = get_idx(b)
            adj_w[ia].append((ib, length))
            adj_w[ib].append((ia, length))

        return np.array(coords, dtype=np.float64), adj_w
    
    @staticmethod 
    def _build_dual_adjacency(segments): 
        '''
        Nodes are roads and edges are intersections between roads. Dual to the Primal graph which 
        is constructed in the opposite manner 
        '''

        n = len(segments)
        node_to_segments = {}
        for i, (a, b, _, _) in enumerate(segments): 

            node_to_segments.setdefault(a, []).append(i)
            node_to_segments.setdefault(b, []).append(i)

        adj = [set() for _ in range(n)]
        for segs in node_to_segments.values(): 
            if len(segs) < 2: 
                continue 
            for i in range(len(segs)): 
                u = segs[i]
                for j in range(i + 1, len(segs)): 
                    v = segs[j]
                    adj[u].add(v)
                    adj[v].add(u)

    @staticmethod 
    def _approx_betweenness(adj, rng): 
        '''
        Picks some k pivots to compute betweenness on drastically reducing computation time 
        '''

        n = len(adj)
        if n == 0: 
            return np.zeros(0, dtype=np.float64)
        
        k_pivots = max(1, int(np.sqrt(n)))
        pivots   = rng.choice(n, size=min(k_pivots, n), replace=False)
        bc       = np.zeros(n, dtype=np.float64)

        for s in pivots: 
            stack = []
            pred  = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=np.float64)
            dist  = -np.ones(n, dtype=np.int32)

            sigma[s] = 1.0 
            dist[s]  = 0 
            q = deque([s])

            while q: 
                v = q.popleft() 
                stack.append(v)
                for w in adj[v]: 
                    if dist[w] < 0: 
                        dist[w] = dist[v] + 1 
                        q.append(w)
                    if dist[w] == dist[v] + 1: 
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta = np.zeros(n, dtype=np.float64)
            while stack: 
                w = stack.pop() 
                if sigma[w] == 0: 
                    continue 
                for v in pred[w]: 
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s: 
                    bc[w] += delta[w]

        bc /= float(len(pivots))
        return bc 

    @staticmethod 
    def _dijkstra(adj_w, source): 
        n = len(adj_w)
        dist = [np.inf] * n 
        dist[source] = 0.0 
        heap = [(0.0, source)]

        while heap: 
            d, u = heapq.heappop(heap)
            if d != dist[u]: 
                continue 

            for v, w in adj_w[u]: 
                nd = d + w 
                if nd < dist[v]: 
                    dist[v] = nd 
                    heapq.heappush(heap, (nd, v))
        
        return dist 

    @staticmethod 
    def _straightness_mean(segments, rng): 
        '''
        Measure of mean curvature of rodes 
        '''
        coords, adj_w = TigerNHCS._build_primal_graph(segments) 
        n = len(coords)
        if n < 2: 
            return 0.0 

        k_pivots = max(1, int(np.sqrt(n)))
        pivots   = rng.choice(n, size=min(k_pivots, n), replace=False)

        ratios = []
        for source in pivots:
            dist = TigerNHCS._dijkstra(adj_w, source)
            sx, sy = coords[source]
            for target in range(n): 
                if target == source: 
                    continue 
                d = dist[target]
                if not np.isfinite(d) or d <= 0: 
                    continue 
                tx, ty = coords[target]
                euclid = np.hypot(sx - tx, sy - ty)
                if euclid > 0: 
                    ratios.append(euclid / d)

        return float(np.mean(ratios)) if ratios else 0.0 

    @staticmethod 
    def _orientation_entropy(segments, bins=36): 
        '''
        Measurement of randomness in orientation for roads, 
        lower entropy indicates better planned development or just higher development in general 
        '''

        if not segments: 
            return 0.0 

        lengths  = np.array([s[2] for s in segments], dtype=np.float64)
        bearings = np.array([s[3] for s in segments], dtype=np.float64)

        total = lengths.sum() 
        if total <= 0: 
            return 0.0 

        bin_width = 180.0 / bins 
        idx = np.floor(bearings / bin_width).astype(np.int32)
        idx = np.clip(idx, 0, bins - 1) 
        weights = np.bincount(idx, weights=lengths, minlength=bins)
        p = weights / total 
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    @staticmethod 
    def _integration_mean(adj, radius=3): 
        if not adj: 
            return 0.0 
        inv_mean = []
        for source in range(len(adj)): 
            dist = {source: 0}
            q = deque([source])

            while q: 
                v = q.popleft() 
                if dist[v] >= radius: 
                    continue 
                for w in adj[v]: 
                    if w not in dist: 
                        dist[w] = dist[v] + 1 
                        q.append(w)
            
            depths = [d for n, d in dist.items() if n != 5 and d <= radius]
            if not depths: 
                continue 

            mean_depth = sum(depths) / len(depths)
            inv_mean.append(1.0 / (mean_depth + 1e-6))

        return float(np.mean(inv_mean)) if inv_mean else 0.0 


def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("--gdb", required=True)
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default=project_path("data", "nchs", "nchs_classification.csv"))
    args = parser.parse_args()
    
    TigerNHCS(args.gdb, args.shapefile, args.out, args.labels).run()


if __name__ == "__main__":
    main() 
