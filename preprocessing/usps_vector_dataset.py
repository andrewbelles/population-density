#!/usr/bin/env python3 
# 
# usps_vector_dataset.py  Andrew Belles  Jan 28th, 2026 
# 
# Generates the master vector file for which the image tensor dataset will be derived 
# 
# 

import argparse 

import numpy as np 

import geopandas as gpd 

import pandas as pd 

from pathlib import Path 


def find_tract_files(root: Path) -> list[Path]: 
    return sorted(root.rglob("*tract*.shp"))

def load_usps_attrs(path: Path) -> pd.DataFrame: 
    gdf  = gpd.read_file(path)
    df   = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df.columns = [c.lower() for c in df.columns]

    df["geoid"] = df["geoid"].astype(str).str.zfill(11)
    df   = df.rename(columns={"geoid": "GEOID"})

    cols = [
        "GEOID",
        "ams_res", 
        "ams_bus",
        "res_vac", 
        "bus_vac", 
        "nostat_res",
        "nostat_bus",
        "nostat_oth"
    ]

    missing = [c for c in cols if c not in df.columns]
    if missing: 
        raise ValueError(f"missing USPS columns: {missing}")

    df = df[cols].copy() 
    for c in cols[1:]: 
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=cols[1:])
    return df 

def compute_channels(df: pd.DataFrame) -> pd.DataFrame: 

    df["no_stat"] = df["nostat_res"] + df["nostat_bus"] + df["nostat_oth"]

    total = (df["ams_res"] + df["ams_bus"] + df["res_vac"] + df["bus_vac"] +
             df["no_stat"])

    df = df[total > 0].copy() 
    
    df["total_addresses"] = total 
    df["comm_ratio"] = df["ams_bus"] / total 
    df["vac_rate"]   = df["res_vac"] / total 

    bus_total = df["ams_bus"] + df["bus_vac"]
    df["bus_vac_rate"] = np.where(bus_total > 0, df["bus_vac"] / bus_total, 0.0)

    df["flux_rate"] = df["no_stat"] / total 

    return df 


def main(): 

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tracts-root", type=Path, required=True)
    parser.add_argument("--usps-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    args   = parser.parse_args()

    tract_files = find_tract_files(args.tracts_root)
    if not tract_files: 
        raise FileNotFoundError(f"No tract shapefiles found under {args.tracts_root}")

    print(f"[info] found {len(tract_files)} tract shapefiles")
    tracts = gpd.GeoDataFrame(pd.concat(
        [gpd.read_file(p)[["GEOID", "geometry"]] for p in tract_files],
        ignore_index=True
    ))

    usps = load_usps_attrs(args.usps_path)
    usps = compute_channels(usps)

    merged = tracts.merge(
        usps[["GEOID", "flux_rate", "bus_vac_rate", "comm_ratio", "vac_rate", "total_addresses"]],
        on="GEOID",
        how="inner"
    )

    print(f"[info] writing {args.out_path}")
    merged.to_file(args.out_path, driver="GPKG")


if __name__ == "__main__":
    main() 
