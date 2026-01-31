#!/usr/bin/env python3 
# 
# saipe_nchs_dataset.py  Andrew Belles  Jan 11th, 2026 
# 
# Loads SAIPE socioeconomic and poverty data labeled against NCHS urban-rural classification 
# 
# 

import argparse, re 

import numpy as np 

import pandas as pd 

from pathlib import Path 

from scipy.io import savemat 

from utils.helpers import project_path


class SaipeNCHS: 

    def __init__(
        self,
        csv_path: str | None = None, 
        labels_path: str | None = None 
    ): 
        if csv_path is None: 
            csv_path = project_path("data", "saipe", "saipe_2013.csv")
        if labels_path is None: 
            labels_path = project_path("data", "nchs", "nchs_classification_2013.csv")

        self.csv_path    = csv_path 
        self.labels_path = labels_path
        self.labels_map  = self._load_labels()
        self.data        = self._build()

    def save(self, out_path: str): 
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        savemat(out, self.data)

    def _build(self): 
        df   = pd.read_csv(self.csv_path, header=None, dtype=str)
        row0 = df.iloc[0].tolist() 
        row1 = df.iloc[1].tolist() 

        feature_names, col_idx = self._build_feature_names(row0, row1)

        data   = df.iloc[2:].reset_index(drop=True)
        state  = data.iloc[:, 0].astype(str).str.zfill(2)
        county = data.iloc[:, 1].astype(str).str.zfill(3)

        mask   = (
            state.str.fullmatch(r"\d{2}")
            & county.str.fullmatch(r"\d{3}")
            & (state != "00")
            & (county != "000")
        )

        data   = data[mask].reset_index(drop=True)
        state  = state[mask].reset_index(drop=True)
        county = county[mask].reset_index(drop=True)
        fips   = (state + county).astype(str)

        X = data.iloc[:, col_idx].copy() 
        X = X.replace({",": ""}, regex=True)
        X = X.mask(X == ".")
        X = X.apply(pd.to_numeric, errors="coerce")

        keep = X.notna().any(axis=0).to_numpy() 
        X    = X.loc[:, keep]
        feature_names = np.asarray([n for n, k in zip(feature_names, keep) if k], dtype="U64")

        row_keep = X.notna().all(axis=1)
        if not row_keep.all(): 
            data = data[row_keep].reset_index(drop=True)
            fips = fips[row_keep].reset_index(drop=True)
            X    = X.loc[row_keep].reset_index(drop=True)

        keep_mask = fips.isin(self.labels_map)
        fips      = fips[keep_mask].to_numpy(dtype="U5")
        X         = X.loc[keep_mask].to_numpy(dtype=np.float64)
        y         = np.array([self.labels_map[f] for f in fips], dtype=np.int64).reshape(-1, 1)

        return {
            "features": X,
            "labels": y,
            "fips_codes": fips,
            "feature_names": feature_names
        }

    def _load_labels(self) -> dict[str, int]: 
        path = Path(self.labels_path)
        if not path.exists(): 
            raise FileNotFoundError(f"label CSV not found: {path}")

        df       = pd.read_csv(path, dtype=str)
        required = {"FIPS", "class_code"}
        if not required.issubset(df.columns): 
            raise ValueError(f"label CSV missing headers: {required}")

        labels = {}
        for _, row in df.iterrows():
            fips = str(row.get("FIPS", "")).strip().zfill(5)
            code = str(row.get("class_code", "")).strip() 
            if fips and code.isdigit(): 
                labels[fips] = int(code) - 1

        if not labels: 
            raise ValueError("label map is empty")
        return labels 

    @staticmethod 
    def _state_slug(text: str) -> str: 
        s = text.strip().lower() 
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    @staticmethod 
    def _metric_name(raw: str) -> str: 
        s    = raw.strip()
        base = s.split(",")[0].strip() 
        return {
            "Poverty Estimate": "poverty_estimate",
            "Poverty Percent": "poverty_percent",
            "90% CI Lower Bound": "ci90_lower", 
            "90% CI Upper Bound": "ci90_upper", 
            "Median Household Income": "median_household_income"
        }.get(base, SaipeNCHS._state_slug(base))

    @staticmethod 
    def _build_feature_names(row0, row1): 
        groups = []
        last   = None 
        for g in row0: 
            if isinstance(g, str) and g.strip(): 
                last = g.strip() 
            groups.append(last)

        names   = []
        col_idx = []
        for i, sub in enumerate(row1): 
            if i < 4: 
                continue 
            group_raw = groups[i] or "feature"
            group     = SaipeNCHS._state_slug(str(group_raw))
            metric    = SaipeNCHS._metric_name(str(sub))
            name      = f"{group}__{metric}" if group != metric else metric 
            names.append(name)
            col_idx.append(i)

        return names, col_idx 


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=project_path("data", "saipe", "saipe_2023.csv"))
    parser.add_argument("--labels", default=project_path("data", "nchs", 
                                                         "nchs_classification_2023.csv"))
    parser.add_argument("--out", default=project_path("data", "datasets",
                                                      "saipe_nchs_2023.mat"))
    args = parser.parse_args()

    data = SaipeNCHS(args.csv, args.labels)
    data.save(args.out)

    n = data.data["features"].shape[0]
    d = data.data["features"].shape[1]

    print(f"[saipe] saved {args.out} ({n} rows, {d} features)")


if __name__ == "__main__": 
    main() 
