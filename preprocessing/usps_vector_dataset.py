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
        "ams_oth",
        "res_vac", 
        "bus_vac",
        "oth_vac", 
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

    df = df.dropna(subset=cols[1:], how="all")
    return df 

def compute_channels(df: pd.DataFrame) -> pd.DataFrame: 
    df["total_res"] = (
        df["ams_res"].fillna(0) + 
        df["res_vac"].fillna(0) + 
        df["nostat_res"].fillna(0)
    )
    df["total_business"] = (
        df["ams_bus"].fillna(0) + 
        df["bus_vac"].fillna(0) + 
        df["nostat_bus"].fillna(0)
    )
    df["total_other"] = (
        df["ams_oth"].fillna(0) + 
        df["oth_vac"].fillna(0) + 
        df["nostat_oth"].fillna(0)
    )
    df["total_addresses"] = (
        df["total_res"] + 
        df["total_business"] + 
        df["total_other"]
    )

    df = df[df["total_addresses"] > 0].copy() 
    
    df["comm_ratio"] = df["total_business"] / df["total_addresses"]
    
    total_vac = (
        df["res_vac"].fillna(0) + 
        df["bus_vac"].fillna(0) + 
        df["oth_vac"].fillna(0)
    )
    df["vac_rate"] = total_vac / df["total_addresses"]

    total_nostat = (
        df["nostat_res"].fillna(0) + 
        df["nostat_bus"].fillna(0) + 
        df["nostat_oth"].fillna(0)
    )
    df["flux_rate"] = total_nostat / df["total_addresses"]

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
        usps[[
            "GEOID", 
            "flux_rate", 
            "comm_ratio", 
            "total_other", 
            "total_business", 
            "total_addresses"
        ]],
        on="GEOID",
        how="inner"
    )

    print(f"[info] writing {args.out_path}")
    merged.to_file(args.out_path, driver="GPKG")


if __name__ == "__main__":
    main() 
