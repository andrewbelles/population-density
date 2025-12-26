#!/usr/bin/env python3 
# 
# viirs_nchscode_dataset.py  Andrew Belles  Dec 22nd, 2025 
# 
# Generates Dataset based on VIIRS nighttime light radiance 
# with NCHS Urban-Rural classification labels. 
# 

import csv
from pathlib import Path
from typing import Dict 

import numpy as np 
import pandas as pd 

from scipy.io import savemat 
from scipy.stats import entropy, kurtosis, skew

from rasterstats import zonal_stats 
import fiona 

from support.helpers import project_path

class ViirsDataset: 
    STATS = ["min", "max", "mean"]
    EXTRA = ["variance", "entropy"]

    def __init__(
        self, 
        viirs_path: str | None = None, 
        counties_path: str | None = None, 
        labels_path: str | None = None, 
        *, 
        chunk_size: int = 200, 
        all_touched: bool = False,
        smush: bool = False 
    ): 

        if viirs_path is None: 
            viirs_path = project_path("data", "viirs", "viirs_2023_avg_masked.dat.tif")
        if counties_path is None: 
            counties_path = project_path("data", "geography", "county_shapefile",
                                         "tl_2020_us_county.shp")
        if labels_path is None: 
            labels_path = project_path("data", "nchs", "nchs_classification.csv")

        self.viirs_path    = viirs_path 
        self.counties_path = counties_path 
        self.labels_path   = labels_path

        self.fips_field    = "GEOID"
        self.chunk_size    = int(chunk_size)
        self.all_touched   = bool(all_touched)
        self.smush         = bool(smush)

        self.label_map = self._load_labels() 
        if self.smush:
            self.label_map = {
                fips: self._coarse_label(code) for fips, code in self.label_map.items() 
            }
        self.df = self._build() 

    def _load_labels(self) -> Dict[str, int]: 

        path = Path(self.labels_path)
        if not path.exists():
            raise FileNotFoundError(f"label CSV not found: {path}")

        labels: Dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as f: 
            reader = csv.DictReader(f) 
            if reader.fieldnames is None: 
                raise ValueError("label CSV missing header row")
            if "FIPS" not in reader.fieldnames or "class_code" not in reader.fieldnames: 
                raise ValueError("expected header FIPS,class_code, got {reader.fieldnames}")

            for row in reader: 
                fips = (row.get("FIPS") or "").strip() 
                code = (row.get("class_code") or "").strip() 
                if not fips or not code: 
                    continue 
                if fips.isdigit(): 
                    fips = fips.zfill(5)
                try: 
                    labels[fips] = int(code) - 1
                except ValueError: 
                    continue 

        if not labels: 
            raise ValueError("label map is empty")
        return labels 

    def _build(self) -> pd.DataFrame: 
        viirs_path    = Path(self.viirs_path)
        counties_path = Path(self.counties_path)

        if not viirs_path.exists(): 
            raise FileNotFoundError(f"VIIRS raster not found: {viirs_path}")
        if not counties_path.exists(): 
            raise FileNotFoundError(f"county shapefile not found: {counties_path}")

        rows = []

        custom_stats_map = {
            "variance": self._get_variance, 
            "entropy": self._get_entropy, 
            "skew": self._get_skew, 
            "kurtosis": self._get_kurtosis
        }

        with fiona.open(counties_path) as source: 

            for geoms, fips_list in self._iter_chunks(source):
                stats = zonal_stats(
                    geoms, 
                    str(viirs_path),
                    stats=self.STATS, 
                    add_stats=custom_stats_map,
                    nodata=None,
                    all_touched=self.all_touched, 
                    geojson_out=False 
                )

                all_stats_keys = self.STATS + self.EXTRA 

                for fips, s in zip(fips_list, stats): 
                    # Skip an entire row if NaN value for any stat
                    if s is None: 
                        continue 
                    if any(s.get(k) is None for k in all_stats_keys): 
                        continue 

                    mean_val = float(s["mean"])
                    var_val  = float(s["variance"])

                    if mean_val > 1e-6: 
                        cv = np.sqrt(var_val) / mean_val 
                    else: 
                        cv = 0.0 

                    rows.append({
                        "FIPS": fips, 
                        **{f"viirs_{k}": float(s[k]) for k in all_stats_keys}, 
                        "viirs_cv": cv, 
                        "label": self.label_map[fips]
                    })

        if not rows: 
            raise ValueError("No rows produced.")

        df = (pd.DataFrame(rows)
              .drop_duplicates(subset=["FIPS"])
              .sort_values("FIPS")
              .reset_index(drop=True))
        print(f"> VIIRS counties processed: {len(df)}")
        return df 

    def _iter_chunks(self, source): 
        geoms = []
        fips_list = []
        for feature in source: 
            properties = feature.get("properties", {})
            fips = str(properties.get(self.fips_field, "")).strip() 
        
            if fips.isdigit(): 
                fips = fips.zfill(5)
            if not fips or fips not in self.label_map: 
                continue 

            geoms.append(feature["geometry"])
            fips_list.append(fips) 

            if len(geoms) >= self.chunk_size:
                yield geoms, fips_list 
                geoms, fips_list = [], []

        if geoms: 
            yield geoms, fips_list

    def save(self, output_path: str): 
        if self.df is None or self.df.empty: 
            raise ValueError("No dataset to save")

        feature_cols = [c for c in self.df.columns if c.startswith("viirs_")]
        feature_cols.sort() 

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
    def _get_variance(x): 
        return np.var(x)

    @staticmethod 
    def _get_entropy(x): 
        valid = x.compressed() if hasattr(x, "compressed") else x
        if valid.size == 0: 
            return 0.0 

        counts, _ = np.histogram(valid, bins=100, density=False)
        return entropy(counts)

    @staticmethod 
    def _get_skew(x): 
        valid = x.compressed() if hasattr(x, "compressed") else x 
        return skew(valid) if valid.size > 0 else 0.0 

    @staticmethod
    def _get_kurtosis(x):
        valid = x.compressed() if hasattr(x, "compressed") else x 
        return kurtosis(valid) if valid.size > 0 else 0.0

    @staticmethod
    def _coarse_label(code: int) -> int: 
        if code in (0, 1): 
            return 0 
        if code in (2, 3): 
            return 1 
        if code in (4, 5): 
            return 2 
        raise ValueError(f"unexpected class code: {code}")


def main(): 

    output_path = project_path("data", "datasets", "viirs_nchs_2023.mat")
    dataset = ViirsDataset(smush=False) 
    dataset.save(output_path)


if __name__ == "__main__": 
    main() 
