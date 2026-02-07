#!/usr/bin/env python3 
# 
# usps_scalar_dataset.py  Andrew Belles  Feb 6th 2026 
# 
# Aggregate USPS tract-level metrics to county-level scalars using same channels 
# as the USPS tensor dataset 
# 

import argparse, rasterio  

import rasterio.mask 

from pathlib import Path 

from typing import Dict 

import numpy as np 

import pandas as pd 

import geopandas as gpd 

from scipy.io import savemat 

from scipy.stats import entropy 

from utils.helpers import project_path

from shapely.geometry import shape, box 

from rasterio.warp import transform_geom 


class UspsScalarDataset: 

    CHANNELS = [
        "flux_rate",        # rate of address change  
        "comm_ratio",       # commercial signal 
        "inst_ratio",       # institution signal 
        "coverage_ratio",   # trust signal 
        "log_density_land", # density over sampled land 
        "address_hhi",      # concentration 
        "log_density_iqr"   # variance of density 
    ]

    def __init__(
        self, 
        usps_gpkg: str | None = None, 
        labels_path: str | None = None, 
        counties_path: str | None = None, 
        *, 
        layer: str | None = None, 
        area_crs: str = "EPSG:5070",
    ): 

        if usps_gpkg is None:
            usps_gpkg = project_path("data", "usps", "usps_master_tracks_2013.gpkg")
        if counties_path is None: 
            counties_path = project_path("data", "geography", "county_shapefile",
                                         "tl_2020_us_county.shp")
        
        if labels_path is None: 
            labels_path = project_path("data", "nchs", "nchs_classification_2013.csv")

        self.usps_gpkg     = usps_gpkg 
        self.labels_path   = labels_path 
        self.counties_path = counties_path
        self.layer         = layer 
        self.area_crs      = area_crs
        
        self.label_map     = self.load_labels() 
        self.df            = self.build() 

    def save(self, out_path: str, csv_path: str | None = None): 
        if self.df is None or self.df.empty: 
            raise ValueError("no dataset to save")

        feature_cols = [f"usps_{c}" for c in self.CHANNELS]

        missing = [c for c in feature_cols if c not in self.df.columns]
        if missing: 
            raise ValueError(f"misiing computed columns: {missing}")

        savemat(out_path, {
            "features": self.df[feature_cols].to_numpy(dtype=np.float64),
            "labels": self.df["label"].to_numpy(dtype=np.int64).reshape(-1, 1), 
            "feature_names": np.array(feature_cols, dtype="U"), 
            "fips_codes": self.df["fips"].to_numpy(dtype="U5"), 
            "n_counties": self.df.shape[0]
        })

        print(f"Saved .mat file: {out_path} ({self.df.shape[0]} rows)")

        if csv_path: 
            self.df.to_csv(csv_path, index=False)
            print(f"Saved .csv file: {csv_path} ({self.df.shape[0]} rows)")

    def build(self) -> pd.DataFrame: 
        gpkg_path = Path(self.usps_gpkg)
        if not gpkg_path.exists(): 
            raise FileNotFoundError(f"missing USPS gpkg: {gpkg_path}")

        gdf = gpd.read_file(gpkg_path, layer=self.layer)
        gdf.columns = [c.lower() for c in gdf.columns]

        required = ["geoid", "total_addresses", "total_business", "total_other", "flux_rate"] 
        missing  = [c for c in required if c not in gdf.columns]
        if missing: 
            raise ValueError(f"USPS gpkg missing columns: {missing}")
        
        gdf["geoid"] = gdf["geoid"].astype(str).str.zfill(11)
        gdf["fips"]  = gdf["geoid"].str[:5]
        gdf = gdf[gdf["fips"].isin(self.label_map.keys())].copy() 

        if gdf.crs.to_string() != self.area_crs: 
            gdf = gdf.to_crs(self.area_crs)

        gdf["tract_area_sqkm"] = gdf.geometry.area / 1e6 
        gdf = gdf[gdf["tract_area_sqkm"] > 1e-6].copy() 

        counties_path = Path(self.counties_path)
        if not counties_path.exists(): 
            raise FileNotFoundError(f"missing county shapefile: {counties_path}")

        counties = gpd.read_file(counties_path)
        if "GEOID" not in counties.columns or "ALAND" not in counties.columns: 
            raise ValueError("county shapefile missing GEOID or ALAND")

        county_area_map = {
            str(r["GEOID"]).zfill(5): float(r["ALAND"]) / 1e6 for _, r in counties.iterrows()
        }

        def agg(group: pd.DataFrame) -> pd.Series: 
            fips = group["fips"].iloc[0]

            s_addr = group["total_addresses"].sum()
            s_bus  = group["total_business"].sum() 
            s_oth  = group["total_other"].sum() 

            s_covered_area = group["tract_area_sqkm"].sum() 
            s_land_area    = county_area_map.get(fips, s_covered_area)

            s_land_area    = max(s_land_area, 1e-6)
            s_covered_area = max(s_covered_area, 1e-6)
            s_addr_        = max(s_addr, 1.0)
            
            out = {}

            if s_addr > 0: 
                out["usps_flux_rate"]  = np.average(
                    group["flux_rate"], 
                    weights=group["total_addresses"]
                )
                out["usps_comm_ratio"] = s_bus / s_addr_ 
                out["usps_inst_ratio"] = s_oth / s_addr_
            else: 
                out["usps_flux_rate"]  = 0.0
                out["usps_comm_ratio"] = 0.0 
                out["usps_inst_ratio"] = 0.0 

            out["usps_coverage_ratio"] = min(s_covered_area / s_land_area, 1.05) 
            
            raw_dens_land = s_addr_ / s_land_area 
            out["usps_log_density_land"] = np.log1p(raw_dens_land)

            # sum of squared shares of addresses 
            shares = group["total_addresses"] / s_addr_ 
            out["usps_address_hhi"] = (shares**2).sum() 
    
            # variance of density regimes within county 
            local_dens = group["total_addresses"] / group["tract_area_sqkm"] 
            local_log  = np.log1p(local_dens) 

            if len(local_log) > 1: 
                q75, q25 = np.percentile(local_log, [75, 25]) 
                out["usps_log_density_iqr"] = q75 - q25 
            else: 
                out["usps_log_density_iqr"] = 0.0 

            return pd.Series(out)
        
        df = gdf.groupby("fips", as_index=False).apply(agg) 
        if isinstance(df, pd.DataFrame) and "fips" not in df.columns: 
            df = df.reset_index()

        df["label"] = df["fips"].map(self.label_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        feature_cols = [f"usps_{c}" for c in self.CHANNELS]
        df = df[["fips", "label", *feature_cols]]
        df = df.sort_values("fips").reset_index(drop=True)

        if df.empty:
            raise ValueError("no rows produced")
        return df 

    def load_labels(self) -> Dict[str, int]: 
        path = Path(self.labels_path)
        if not path.exists(): 
            raise FileNotFoundError(f"missing labels CSV: {path}")

        labels: Dict[str, int] = {}
        df = pd.read_csv(path, dtype=str)

        required = {"FIPS", "class_code"}
        if not required.issubset(df.columns): 
            raise ValueError(f"label CSV expected: {required}")

        for _, row in df.iterrows(): 
            fips = str(row.get("FIPS", "")).strip().zfill(5)
            code = str(row.get("class_code", "")).strip() 
            if fips and code.isdigit(): 
                labels[fips] = int(code) - 1 

        if not labels: 
            raise ValueError("label map is empty")
        return labels 


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--usps-gpkg", type=str, required=True)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--counties-path", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--area-crs", type=str, default="EPSG:5070")
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, default=None)
    args = parser.parse_args()

    ds = UspsScalarDataset(
        usps_gpkg=args.usps_gpkg,
        labels_path=args.labels_path,
        counties_path=args.counties_path,
        layer=args.layer,
        area_crs=args.area_crs,
    )
    ds.save(args.out_path, csv_path=args.csv_path)


if __name__ == "__main__": 
    main() 
