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

    CHANNELS = ["flux_rate", "texture", "comm_ratio", "vac_rate"]

    def __init__(
        self, 
        usps_gpkg: str | None = None, 
        labels_path: str | None = None, 
        nlcd_path: str | None = None, 
        *, 
        layer: str | None = None, 
        area_crs: str = "EPSG:5070",
        water_codes: tuple[int, ...] = (11, 12) 
    ): 

        if usps_gpkg is None:
            usps_gpkg = project_path("data", "usps", "usps_master_tracks_2013.gpkg")
        if labels_path is None: 
            labels_path = project_path("data", "nchs", "nchs_classification_2013.csv")
        if nlcd_path is None: 
            nlcd_path = project_path("data", "nlcd", "Annual_NLCD_LndCov_2013_CU_C1V1.tif")

        self.usps_gpkg   = usps_gpkg 
        self.labels_path = labels_path 
        self.nlcd_path   = nlcd_path 
        self.layer       = layer 
        self.area_crs    = area_crs
        self.water_codes = tuple(int(x) for x in water_codes)
        
        self.label_map   = self.load_labels() 
        self.df          = self.build() 

    def save(self, out_path: str, csv_path: str | None = None): 
        if self.df is None or self.df.empty: 
            raise ValueError("no dataset to save")

        feature_cols = [f"usps_{c}" for c in self.CHANNELS]
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
        gdf["texture"] = self.compute_texture(gdf)

        required = ["geoid"] + self.CHANNELS 
        missing  = [c for c in required if c not in gdf.columns]
        if missing: 
            raise ValueError(f"USPS gpkg missing columns: {missing}")
        
        gdf["geoid"] = gdf["geoid"].astype(str).str.zfill(11)
        gdf["fips"]  = gdf["geoid"].str[:5]

        gdf = gdf[gdf["fips"].isin(self.label_map.keys())].copy() 
        gdf = gdf.dropna(subset=self.CHANNELS)

        if gdf.crs is None: 
            raise ValueError("USPS gpkg missing CRS, can't compute area weights")

        area = gdf.to_crs(self.area_crs).geometry.area 
        gdf["area"] = area 
        gdf  = gdf[gdf["area"] > 0].copy() 

        def agg(group: pd.DataFrame) -> pd.Series: 
            w = group["area"].to_numpy()
            den = w.sum() 
            if den <= 0:
                 return pd.Series({f"usps_{c}": np.nan for c in self.CHANNELS})
            out = {}
            for c in self.CHANNELS:
                out[f"usps_{c}"] = np.average(group[c].to_numpy(), weights=w)
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

    def compute_texture(self, gdf): 
        if gdf.crs is None:
            raise ValueError("GPKG missing CRS; cannot reproject")

        textures = []
        with rasterio.open(self.nlcd_path) as src:
            raster_bounds = box(*src.bounds)
            src_crs = src.crs
            gdf_crs = gdf.crs

            for geom in gdf.geometry:
                if geom is None:
                    textures.append(np.nan)
                    continue

                g = geom
                if gdf_crs and src_crs and gdf_crs != src_crs:
                    g = shape(transform_geom(
                        gdf_crs, 
                        src_crs, 
                        geom.__geo_interface__, 
                        precision=6)
                    )

                if not g.intersects(raster_bounds):
                    textures.append(np.nan)
                    continue

                try:
                    out, _ = rasterio.mask.mask(src, [g], crop=True)
                except ValueError:
                    textures.append(np.nan)
                    continue

                data = out[0]
                nodata = src.nodata if src.nodata is not None else 0
                valid = (data != nodata)
                if valid.sum() == 0:
                    textures.append(np.nan)
                    continue

                vals = data[valid]
                if self.water_codes:
                    vals = vals[~np.isin(vals, self.water_codes)]
                if vals.size == 0:
                    textures.append(0.0)
                    continue

                _, counts = np.unique(vals, return_counts=True)
                textures.append(float(entropy(counts)))

        return textures


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--usps-gpkg", type=str, required=True)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--nlcd-path", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--area-crs", type=str, default="EPSG:5070")
    parser.add_argument("--water-codes", nargs="*", type=int, default=[11, 12])
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, default=None)
    args = parser.parse_args()

    ds = UspsScalarDataset(
        usps_gpkg=args.usps_gpkg,
        labels_path=args.labels_path,
        nlcd_path=args.nlcd_path,
        layer=args.layer,
        area_crs=args.area_crs,
        water_codes=args.water_codes
    )
    ds.save(args.out_path, csv_path=args.csv_path)


if __name__ == "__main__": 
    main() 
