#!/usr/bin/env python3
#
# tiger_nchs_dataset.py  Andrew Belles  Dec 23rd, 2025  
#
# Processes road data on a per county basis from TIGER dataset
#

import argparse, os, fiona, time
import geopandas as gpd
import pandas as pd
import numpy as np 
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
from scipy.io import savemat 
from multiprocessing import Pool, cpu_count
from support.helpers import project_path

# Global variables for workers (avoids pickling large objects)
WORKER_GDF = None
WORKER_SINDEX = None
WORKER_TRANSFORMER = None
WORKER_MTFCC_HWY = None
WORKER_MTFCC_DRIVE = None
WORKER_LABEL_MAP = None

def init_worker(county_shp, src_crs, proj_crs, label_map):
    """
    Initializes each worker process.
    Loads the county shapefile ONCE per core so we don't pass it over IPC.
    """
    global WORKER_GDF, WORKER_SINDEX, WORKER_TRANSFORMER
    global WORKER_MTFCC_HWY, WORKER_MTFCC_DRIVE, WORKER_LABEL_MAP
    
    # 1. Load Counties
    # print(f"[Worker {os.getpid()}] Loading counties...")
    gdf = gpd.read_file(county_shp).to_crs(src_crs)
    gdf["FIPS"] = gdf["GEOID"]
    
    # Filter by labels (optimization)
    gdf = gdf[gdf["FIPS"].isin(label_map.keys())]
    
    WORKER_GDF = gdf
    WORKER_SINDEX = gdf.sindex
    WORKER_LABEL_MAP = label_map
    
    # 2. Setup Transformers & Constants
    WORKER_TRANSFORMER = Transformer.from_crs(src_crs, proj_crs, always_xy=True).transform
    
    # Constants
    WORKER_MTFCC_HWY = {'S1100', 'S1200'}
    WORKER_MTFCC_DRIVE = {'S1100', 'S1200', 'S1400'}

def process_batch(features_batch):
    """
    Pure CPU task: Takes a list of raw feature dicts, performs spatial operations.
    Returns a dictionary of aggregated stats for the batch.
    """
    # Local aggregation for this batch
    # FIPS -> {len_hwy, len_local, nodes: {}}
    local_stats = {}
    
    for props, geom_obj in features_batch:
        mtfcc = props.get('MTFCC', '')
        
        # 1. Spatial Lookup (Lat/Lon)
        # Using the global worker sindex
        possible_idxs = list(WORKER_SINDEX.query(geom_obj.centroid, predicate='intersects'))
        
        if not possible_idxs:
            continue
            
        matched_fips = WORKER_GDF.iloc[possible_idxs[0]]["FIPS"]
        
        # Initialize if new for this batch
        if matched_fips not in local_stats:
            local_stats[matched_fips] = {"len_hwy": 0.0, "len_local": 0.0, "nodes": {}}
        
        stats = local_stats[matched_fips]
        
        # 2. Project & Measure
        geom_proj = transform(WORKER_TRANSFORMER, geom_obj)
        length_m = geom_proj.length
        
        if mtfcc in WORKER_MTFCC_HWY:
            stats["len_hwy"] += length_m
        else:
            stats["len_local"] += length_m
            
        # 3. Topology
        # Extract endpoints logic inline for speed
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
    BATCH_SIZE = 20000 # Tune this: Larger = less IPC overhead, more RAM
    
    def __init__(self, gdb_path, county_shp, out_path, labels_path=None, state_filter=None):
        self.gdb_path = gdb_path
        self.county_shp = county_shp
        self.out_path = out_path
        self.state_filter = state_filter
        
        # Load labels in main process to pass to workers
        if labels_path is None:
             # Adjust this import path if needed based on where you run it
             labels_path = "data/nchs/nchs_classification.csv"
        
        self.label_map = self._load_labels(labels_path)

    def _load_labels(self, path):
        labels = {}
        if not os.path.exists(path):
            # Fallback or empty if not found, but better to fail early
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
        
        # 1. Setup Multiprocessing Pool
        n_cores = max(1, cpu_count() - 1) # Leave one for OS/Main
        print(f"[INFO] Initializing Pool with {n_cores} workers...")
        
        pool = Pool(
            processes=n_cores,
            initializer=init_worker,
            initargs=(self.county_shp, self.SRC_CRS, self.PROJ_CRS, self.label_map)
        )

        # 2. Stream Data
        target_layer = self._find_edges_layer()
        print(f"[INFO] Streaming from GDB Layer: {target_layer}")
        
        batch = []
        jobs = []
        
        # Allow filtering driving roads early to reduce IPC payload
        DRIVING = {'S1100', 'S1200', 'S1400'}
        
        with fiona.open(self.gdb_path, layer=target_layer) as src:
            for feature in src:
                props = feature['properties']
                if props.get('MTFCC') not in DRIVING:
                    continue
                
                # Convert to shapely object HERE (lightweight) or inside worker
                # Converting here allows us to filter empty geometries early
                try:
                    geom = shape(feature['geometry'])
                    if not geom.is_empty:
                        batch.append((props, geom))
                except:
                    continue

                if len(batch) >= self.BATCH_SIZE:
                    jobs.append(pool.apply_async(process_batch, (batch,)))
                    batch = []
                    
                    if len(jobs) % 10 == 0:
                        print(f"> Submitted {len(jobs) * self.BATCH_SIZE} roads...", end="\r")

            # Final batch
            if batch:
                jobs.append(pool.apply_async(process_batch, (batch,)))

        print(f"\n[INFO] All tasks submitted. Waiting for results...")
        pool.close()
        pool.join()

        # 3. Aggregate Results (Reduce)
        print("[INFO] Aggregating worker results...")
        final_stats = {} # FIPS -> {len_hwy, len_local, nodes: {}}
        
        for job in jobs:
            batch_result = job.get() # Blocking get
            for fips, data in batch_result.items():
                if fips not in final_stats:
                    final_stats[fips] = {"len_hwy": 0.0, "len_local": 0.0, "nodes": {}}
                
                target = final_stats[fips]
                target["len_hwy"] += data["len_hwy"]
                target["len_local"] += data["len_local"]
                
                # Merge nodes (expensive but necessary)
                for k, v in data["nodes"].items():
                    target["nodes"][k] = target["nodes"].get(k, 0) + v

        # 4. Final Dataframe Build (Same as before)
        self._save(final_stats)
        print(f"[DONE] Total time: {time.time() - start_time:.2f}s")

    def _save(self, stats):
        # ... (Same logic as previous script to load county ALAND and save .mat) ...
        # Reloading counties briefly to get ALAND and ensure all FIPS are present
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
            
            # Calculate Ratios
            deg3, deg4 = 0, 0
            for _, count in data["nodes"].items():
                # Node aggregation across batches creates exact counts
                if count == 3: deg3 += 1
                elif count >= 4: deg4 += 1
            
            total = deg3 + deg4
            ratio = deg4 / total if total > 0 else 0.0
            
            rows.append({
                "FIPS": fips,
                "label": self.label_map[fips],
                "tiger_density_hwy": (data["len_hwy"]/1000)/area_km2,
                "tiger_density_local": (data["len_local"]/1000)/area_km2,
                "tiger_ratio_4way": ratio
            })
            
        df = pd.DataFrame(rows)
        if not df.empty:
            feature_cols = ["tiger_density_hwy", "tiger_density_local", "tiger_ratio_4way"]
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
    label_file = project_path("data", "nchs", "nchs_classification.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gdb", required=True)
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default=label_file)
    args = parser.parse_args()
    
    proc = TigerProcessorMulti(args.gdb, args.shapefile, args.out, args.labels)
    proc.run()


if __name__ == "__main__":
    main() 
