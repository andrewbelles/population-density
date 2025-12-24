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

# Global variables for workers
WORKER_GDF = None
WORKER_SINDEX = None
WORKER_TRANSFORMER = None
WORKER_MTFCC_HWY = None
WORKER_LABEL_MAP = None

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
    local_stats = {}
    
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
                "nodes": {}
            }
        
        stats = local_stats[matched_fips]
        
        # 2. Project & Measure
        geom_proj = transform(WORKER_TRANSFORMER, geom_obj)
        length_m = geom_proj.length
        
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

class TigerProcessorMulti:
    
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
                        "nodes": {}
                    }
                
                t = final_stats[fips]
                t["len_hwy"] += data["len_hwy"]
                t["cnt_hwy"] += data["cnt_hwy"]
                t["len_local"] += data["len_local"]
                t["cnt_local"] += data["cnt_local"]
                t["euclid_local"] += data["euclid_local"]
                
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
                "tiger_circuity_local": circuity
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
                "tiger_meshedness"
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


def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("--gdb", required=True)
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default=project_path("data", "nchs", "nchs_classification.csv"))
    args = parser.parse_args()
    
    TigerProcessorMulti(args.gdb, args.shapefile, args.out, args.labels).run()


if __name__ == "__main__":
    main() 
