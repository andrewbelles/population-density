#!/usr/bin/env python3
#
# usps_scalar_dataset.py  Andrew Belles  Feb 6th 2026
#
# Aggregate USPS tract-level metrics to county-level scalars, then append
# administrative/economic texture features from saipe_scalar_{year}.mat.
#

import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.io import loadmat, savemat

from utils.helpers import _mat_str_vector, project_path
from preprocessing.labels import build_label_map


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

    ADMIN_TEXTURE_FEATURES = [
        "median_household_income__value",
        "pov_uncertainty",
        "pov_cv",
        "pov_gap_norm",
        "pov_threeway_entropy",
        "wage_dependence_rate",
        "dividend_rate",
        "unemployment_rate",
        "gig_self_employed_rate",
        "retirement_rate",
        "pov_u18_minus_517_norm",
        "irs_div_plus_ret_per_nonwage",
    ]

    def __init__(
        self,
        usps_gpkg: str | None = None,
        counties_path: str | None = None,
        admin_mat_path: str | None = None,
        *,
        label_year: int = 2013,
        census_dir: str | Path = project_path("data", "census"),
        layer: str | None = None,
        area_crs: str = "EPSG:5070",
    ):

        if usps_gpkg is None:
            usps_gpkg = project_path("data", "usps", "usps_master_tracts_2013.gpkg")
        if counties_path is None:
            counties_path = project_path(
                "data", "geography", "county_shapefile", "tl_2020_us_county.shp"
            )
        if admin_mat_path is None:
            admin_mat_path = project_path("data", "datasets", f"saipe_scalar_{label_year}.mat")

        self.usps_gpkg = usps_gpkg
        self.counties_path = counties_path
        self.admin_mat_path = admin_mat_path
        self.layer = layer
        self.area_crs = area_crs

        self.label_map = build_label_map(label_year, census_dir=census_dir)
        self.df = self.build()

    def save(self, out_path: str, csv_path: str | None = None):
        if self.df is None or self.df.empty:
            raise ValueError("no dataset to save")

        feature_cols = [c for c in self.df.columns if c not in ("fips", "label")]
        if not feature_cols:
            raise ValueError("no feature columns to save")

        savemat(out_path, {
            "features": self.df[feature_cols].to_numpy(dtype=np.float64),
            "labels": self.df["label"].to_numpy(dtype=np.float64).reshape(-1, 1),
            "feature_names": np.array(feature_cols, dtype="U"),
            "fips_codes": self.df["fips"].to_numpy(dtype="U5"),
            "n_counties": self.df.shape[0],
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
        missing = [c for c in required if c not in gdf.columns]
        if missing:
            raise ValueError(f"USPS gpkg missing columns: {missing}")

        gdf["geoid"] = gdf["geoid"].astype(str).str.zfill(11)
        gdf["fips"] = gdf["geoid"].str[:5]
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

        def _group_fips(name) -> str: 
            if isinstance(name, tuple): 
                name = name[0]
            return str(name).strip().zfill(5)

        def agg(group: pd.DataFrame) -> pd.Series:
            fips = _group_fips(group.name)

            s_addr = group["total_addresses"].sum()
            s_bus = group["total_business"].sum()
            s_oth = group["total_other"].sum()

            s_covered_area = group["tract_area_sqkm"].sum()
            s_land_area = county_area_map.get(fips, s_covered_area)

            s_land_area = max(s_land_area, 1e-6)
            s_covered_area = max(s_covered_area, 1e-6)
            s_addr_ = max(s_addr, 1.0)

            out = {}

            if s_addr > 0:
                out["usps_flux_rate"] = np.average(
                    group["flux_rate"],
                    weights=group["total_addresses"],
                )
                out["usps_comm_ratio"] = s_bus / s_addr_
                out["usps_inst_ratio"] = s_oth / s_addr_
            else:
                out["usps_flux_rate"] = 0.0
                out["usps_comm_ratio"] = 0.0
                out["usps_inst_ratio"] = 0.0

            out["usps_coverage_ratio"] = min(s_covered_area / s_land_area, 1.05)

            raw_dens_land = s_addr_ / s_land_area
            out["usps_log_density_land"] = np.log1p(raw_dens_land)

            shares = group["total_addresses"] / s_addr_
            out["usps_address_hhi"] = (shares ** 2).sum()

            local_dens = group["total_addresses"] / group["tract_area_sqkm"]
            local_log = np.log1p(local_dens)

            if len(local_log) > 1:
                q75, q25 = np.percentile(local_log, [75, 25])
                out["usps_log_density_iqr"] = q75 - q25
            else:
                out["usps_log_density_iqr"] = 0.0

            return pd.Series(out)


        grp = gdf.groupby("fips", sort=False)
        df  = grp.apply(agg, include_groups=False).reset_index()
        df["label"] = df["fips"].map(self.label_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(float)

        usps_feature_cols = [f"usps_{c}" for c in self.CHANNELS]
        df = df[["fips", "label", *usps_feature_cols]].copy()
        df = self.merge_admin_texture(df)
        df = df.sort_values("fips").reset_index(drop=True)

        if df.empty:
            raise ValueError("no rows produced")
        return df

    def merge_admin_texture(self, base_df: pd.DataFrame) -> pd.DataFrame:
        p = Path(self.admin_mat_path)
        if not p.exists():
            raise FileNotFoundError(f"missing admin mat file: {p}")

        mat = loadmat(str(p))
        required = {"features", "feature_names", "fips_codes"}
        if not required.issubset(mat):
            raise ValueError(f"{p} missing keys: {sorted(required)}")

        X = np.asarray(mat["features"], dtype=np.float64)
        raw_fips = _mat_str_vector(mat["fips_codes"]).astype("U5")
        raw_names = _mat_str_vector(mat["feature_names"]).astype("U")

        fips  = np.asarray([str(v).strip().zfill(5) for v in raw_fips], dtype="U5")
        names = np.asarray([str(v).strip() for v in raw_names], dtype="U") 

        if X.ndim != 2 or X.shape[0] != fips.shape[0]:
            raise ValueError(f"invalid admin matrix shape: X={X.shape}, fips={fips.shape}")

        name_to_idx = {str(n): i for i, n in enumerate(names.tolist())}
        missing = [c for c in self.ADMIN_TEXTURE_FEATURES if c not in name_to_idx]
        if missing:
            raise ValueError(f"admin texture missing columns: {missing}")

        admin = pd.DataFrame({"fips": fips})
        for c in self.ADMIN_TEXTURE_FEATURES:
            admin[c] = X[:, name_to_idx[c]]

        admin = admin.replace([np.inf, -np.inf], np.nan)
        admin = admin.dropna(subset=self.ADMIN_TEXTURE_FEATURES).reset_index(drop=True)

        out = base_df.merge(admin, on="fips", how="inner")
        if out.empty:
            raise ValueError("no overlap between USPS features and admin texture")
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usps-gpkg", type=str, required=True)
    parser.add_argument("--year", type=int, default=2013)
    parser.add_argument("--census-dir", default=project_path("data", "census"))
    parser.add_argument("--counties-path", type=str, default=None)
    parser.add_argument("--admin-mat", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--area-crs", type=str, default="EPSG:5070")
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, default=None)
    args = parser.parse_args()

    ds = UspsScalarDataset(
        usps_gpkg=args.usps_gpkg,
        counties_path=args.counties_path,
        admin_mat_path=args.admin_mat,
        label_year=args.year,
        census_dir=args.census_dir,
        layer=args.layer,
        area_crs=args.area_crs,
    )
    ds.save(args.out_path, csv_path=args.csv_path)


if __name__ == "__main__":
    main()
