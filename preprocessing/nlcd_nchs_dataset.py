#!/usr/bin/env python3 
# 
# ncld_nchs_dataset.py  Andrew Belles  Dec 24th, 2025 
# 
# Computes NLCD Land Cover statistics and labels against NCHS 
# 

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from scipy.io import savemat
from support.helpers import project_path

import scipy.ndimage as nd 

STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

class NlcdDataset:
    
    # NLCD Class Groups (Annual NLCD Collection 1.0)
    # Group 1: Developed (Urban Strength)
    CODE_DEV_OPEN = 21
    CODE_DEV_LOW  = 22
    CODE_DEV_MED  = 23
    CODE_DEV_HIGH = 24
    
    # Group 2: Agriculture (Rural Economy)
    CODE_AG_PASTURE = 81
    CODE_AG_CROPS   = 82
    
    # Group 3: Nature (Wilderness)
    CODES_NATURE = {41, 42, 43, 52, 71, 90, 95}
    
    # Group 4: Water (Often excluded or used as distinct signal)
    CODES_WATER = {11, 12}

    def __init__(
        self,
        raster_path: str | None = None,
        counties_path: str | None = None,
        labels_path: str | None = None
    ):
        if raster_path is None:
            # UPDATE THIS FILENAME to match your specific download
            raster_path = project_path("data", "nlcd", "Annual_NLCD_LndCov_2023_CU_C1V1.tif")
        if counties_path is None:
            counties_path = project_path("data", "geography", "county_shapefile",
                                         "tl_2020_us_county.shp")
        if labels_path is None:
            labels_path = project_path("data", "nchs", "nchs_classification.csv")

        self.raster_path = raster_path
        self.counties_path = counties_path
        self.labels_path = labels_path
        
        self.label_map = self._load_labels()
        self.df = self._build()

    def _load_labels(self) -> Dict[str, int]:
        path = Path(self.labels_path)
        if not path.exists():
            raise FileNotFoundError(f"Label CSV not found: {path}")

        labels: Dict[str, int] = {}
        # Using pandas for robust CSV reading
        df = pd.read_csv(path, dtype=str)
        
        required = {"FIPS", "class_code"}
        if not required.issubset(df.columns):
             raise ValueError(f"Label CSV missing headers. Expected {required}")

        for _, row in df.iterrows():
            fips = str(row.get("FIPS", "")).strip().zfill(5)
            code = str(row.get("class_code", "")).strip()
            
            if fips and code and code.isdigit():
                labels[fips] = int(code) - 1
        
        if not labels:
            raise ValueError("Label map is empty")
        return labels

    def _build(self) -> pd.DataFrame:
        print(f"[INFO] Loading Counties: {self.counties_path}")
        gdf = gpd.read_file(self.counties_path)
        gdf["FIPS"] = gdf["GEOID"]
        
        # Filter to labeled counties only
        gdf = gdf[gdf["FIPS"].isin(self.label_map.keys())]
        
        print(f"[INFO] Opening NLCD Raster: {self.raster_path}")
        rows = []
        
        with rasterio.open(self.raster_path) as src:
            # Ensure CRS Match
            if gdf.crs != src.crs:
                print(f"[WARN] Reprojecting counties to match NLCD ({src.crs})...")
                gdf = gdf.to_crs(src.crs)
            
            total = len(gdf)
            print(f"[INFO] Processing {total} counties...")

            for idx, row in gdf.iterrows():
                if idx % 50 == 0:
                    print(f"> County {idx}/{total}...", end="\r")
                
                stats = self._process_county(src, row["geometry"])
                if stats:
                    stats["FIPS"] = row["FIPS"]
                    stats["label"] = self.label_map[row["FIPS"]]
                    rows.append(stats)

        print(f"\n[INFO] Processing complete. {len(rows)} valid counties.")
        if not rows:
            raise ValueError("No rows produced.")

        return pd.DataFrame(rows).sort_values("FIPS").reset_index(drop=True)

    def _process_county(self, src, geometry) -> Optional[Dict[str, float]]:
        try:
            # Mask: Crop the raster to the county shape
            # crop=True optimizes the read window
            out_image, _ = rasterio.mask.mask(src, [geometry], crop=True)
            data = out_image[0] # Band 1
            
            # Exclude NoData (usually 0 in NLCD)
            # Use src.nodata if defined, otherwise assume 0
            nodata = src.nodata if src.nodata is not None else 0
            valid_mask   = data != nodata
            valid_pixels = data[valid_mask]
            
            if valid_pixels.size == 0:
                return None
                
            unique, counts = np.unique(valid_pixels, return_counts=True)

            classes = np.array(sorted(unique), dtype=np.int32)
            adj_counts, class_index = self._adjacency_counts(data, valid_mask, classes)

            ai_dev_open = self._aggregation_index(adj_counts, class_index, self.CODE_DEV_OPEN)
            ai_dev_low  = self._aggregation_index(adj_counts, class_index, self.CODE_DEV_LOW)
            ai_dev_med  = self._aggregation_index(adj_counts, class_index, self.CODE_DEV_MED)
            ai_dev_high = self._aggregation_index(adj_counts, class_index, self.CODE_DEV_HIGH)
            contagion   = self._contagion(adj_counts)
            
            edge_dens_dev_open = self._edge_density(
                data == self.CODE_DEV_OPEN, 
                valid_mask
            )
            edge_dens_nature   = self._edge_density(
                np.isin(data, list(self.CODES_NATURE)),
                valid_mask
            )

            lpi_dev_high = self._largest_patch_index(data == self.CODE_DEV_HIGH, valid_mask)

            total_px = valid_pixels.size
            class_counts = dict(zip(unique, counts))
            
            def get_cnt(keys):
                if isinstance(keys, int): keys = [keys]
                return sum(class_counts.get(k, 0) for k in keys)

            # Raw Counts
            c_21 = get_cnt(21) # Open Space (Lawns)
            c_22 = get_cnt(22) # Low Intensity
            c_23 = get_cnt(23) # Med Intensity
            c_24 = get_cnt(24) # High Intensity
            c_dev_total = c_21 + c_22 + c_23 + c_24
            
            # --- METRIC 1: THE LAWN INDEX (Structure) ---
            # Differentiates Suburbs (High) from Cities (Low)
            # Avoid divide by zero if no development
            if c_dev_total > 0:
                lawn_index = c_21 / c_dev_total 
                concrete_index = (c_23 + c_24) / c_dev_total
            else:
                lawn_index = 0.0
                concrete_index = 0.0

            #  EDGE DENSITY
            # Measures how "choppy" the land is.
            # Developed = 21, 22, 23, 24
            bin_dev = np.isin(data, [21, 22, 23, 24]).astype(int)
            
            # Compute edges using a Laplacian filter (finds boundaries)
            edges = nd.laplace(bin_dev)
            
            # Normalize by total area (Edges per Pixel)
            edge_density = np.count_nonzero(edges[valid_mask]) / total_px

            # LANDSCAPE DIVERSITY
            # Measures heterogeneity. Suburbs = High, Rural/Urban = Low
            probs = [c / total_px for c in counts]
            shannon_diversity = -sum(p * np.log(p) for p in probs if p > 0)

            return {
                "nlcd_dev_open": c_21 / total_px,
                "nlcd_dev_low":  c_22 / total_px,
                "nlcd_dev_med":  c_23 / total_px,
                "nlcd_dev_high": c_24 / total_px,
                "nlcd_ag_pasture": get_cnt(self.CODE_AG_PASTURE) / total_px,
                "nlcd_ag_crops":   get_cnt(self.CODE_AG_CROPS) / total_px,
                "nlcd_nature":     get_cnt(list(self.CODES_NATURE)) / total_px,
                "nlcd_lawn_index": lawn_index,
                "nlcd_urban_core": concrete_index,
                "nlcd_edge_dens":  edge_density,
                "nlcd_diversity":  shannon_diversity,
                "nlcd_ai_dev_open": ai_dev_open,
                "nlcd_ai_dev_low": ai_dev_low,
                "nlcd_ai_dev_med": ai_dev_med,
                "nlcd_ai_dev_high": ai_dev_high,
                "nlcd_contagion": contagion,
                "nlcd_edge_dens_dev_open": edge_dens_dev_open,
                "nlcd_edge_dens_nature": edge_dens_nature,
                "nlcd_lpi_dev_high": lpi_dev_high
            }

        except Exception:
            return None

    def save(self, output_path: str):
        if self.df is None or self.df.empty:
            raise ValueError("No dataset to save")
        
        feature_cols = [c for c in self.df.columns if c.startswith("nlcd_")]
        
        mat = {
            "features": self.df[feature_cols].to_numpy(dtype=np.float64),
            "labels": self.df["label"].to_numpy(dtype=np.int64).reshape(-1, 1),
            "feature_names": np.array(feature_cols, dtype="U"),
            "fips_codes": self.df["FIPS"].to_numpy(dtype="U5"),
            "n_counties": self.df.shape[0]
        }

        savemat(output_path, mat)
        print(f"Saved .mat file: {output_path} ({self.df.shape[0]} rows)")

    @staticmethod 
    def _adjacency_counts(data, valid_mask, classes): 
        classes = np.asarray(classes, dtype=np.int32)
        if classes.size == 0: 
            return np.zeros((0, 0), dtype=np.int64), {}
        max_val = int(classes.max())
        lookup  = np.full(max_val + 1, -1, dtype=np.int32)
        lookup[classes] = np.arange(classes.size, dtype=np.int32)

        counts = np.zeros((classes.size, classes.size), dtype=np.int64)

        def add_pairs(a, b, mask): 
            if not np.any(mask): 
                return 
            la = lookup[a[mask]]
            lb = lookup[b[mask]]
            valid = (la >= 0) & (lb >= 0)
            la = la[valid]
            lb = lb[valid]
            np.add.at(counts, (la, lb), 1)
            np.add.at(counts, (lb, la), 1)

        add_pairs(data[:, :-1], data[:, 1:], valid_mask[:, :-1] & valid_mask[:, 1:])
        add_pairs(data[:-1, :], data[1:, :], valid_mask[:-1, :] & valid_mask[1:, :])

        class_index = {int(c): i for i, c in enumerate(classes)}
        return counts, class_index 

    @staticmethod 
    def _aggregation_index(adj_counts, class_index, class_code): 
        idx = class_index.get(int(class_code))
        if idx is None: 
            return 0.0 
        gii = adj_counts[idx, idx]
        gi  = adj_counts[idx, :].sum() 
        if gi == 0:
            return 0.0 
        return float(gii / gi)

    @staticmethod
    def _contagion(adj_counts): 
        total = adj_counts.sum() 
        if total == 0: 
            return 0.0 
        m = adj_counts.shape[0]
        if m <= 1: 
            return 0.0 
        p = adj_counts / total 
        mask = p > 0 
        value = 1.0 + (np.sum(p[mask] * np.log(p[mask])) / (2.0 * np.log(m)))
        return float(value)

    @staticmethod 
    def _edge_density(mask, valid_mask): 
        total = int(valid_mask.sum())
        if total == 0: 
            return 0.0 
        m = mask & valid_mask 
        if not np.any(m): 
            return 0.0 

        edges = 0 
        left  = m[:, :-1]
        right = m[:, 1:]
        edges += np.count_nonzero(left & ~right)
        edges += np.count_nonzero(~left & right)
        up    = m[:-1, :]
        down  = m[1:, :]
        edges += np.count_nonzero(up & ~down)
        edges += np.count_nonzero(~up & down)
        return float(edges / total)

    @staticmethod 
    def _largest_patch_index(mask, valid_mask): 
        total = int(valid_mask.sum())
        if total == 0: 
            return 0.0 
        m = mask & valid_mask 
        if not np.any(m): 
            return 0.0 
        
        labeled, num = nd.label(m, structure=STRUCTURE)
        if num == 0: 
            return 0.0 
        counts = np.bincount(labeled.ravel())
        if counts.size <= 1: 
            return 0.0 
        largest = counts[1:].max()
        return float(largest / total)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", help="Path to NLCD GeoTIFF")
    parser.add_argument("--shapefile", help="Path to County Shapefile")
    parser.add_argument("--labels", help="Path to NCHS Classification CSV")
    parser.add_argument("--out", default=project_path("data", "datasets", "nlcd_nchs_2023.mat"))
    args = parser.parse_args()
    
    # Allow None args to trigger class defaults
    dataset = NlcdDataset(
        raster_path=args.raster,
        counties_path=args.shapefile,
        labels_path=args.labels
    )
    dataset.save(args.out)

if __name__ == "__main__":
    main()
