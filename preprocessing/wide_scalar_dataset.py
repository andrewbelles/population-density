#!/usr/bin/env python3 
# 
# wide_scalar_dataset.py  Andrew Belles  Feb 10th 
# 
# 2-feature Scalar Dataset for wide model 
# - poverty counts from saipe estimates  
# - n2_returns (from IRS csv)
# 

import argparse 

from pathlib import Path 

import pandas as pd 

import numpy as np

from scipy.io import savemat 

from utils.helpers import (
    project_path,
    to_num
)

from preprocessing.labels import build_label_map


YEAR_SPECS = {
    2013: {
        "saipe_csv": project_path("data", "saipe", "saipe_2013.csv"), 
        "ir_csv":    project_path("data", "saipe", "13incyall.csv"),
        "n2_col":    4 
    },
    2019: {
        "saipe_csv": project_path("data", "saipe", "saipe_2019.csv"), 
        "ir_csv":    project_path("data", "saipe", "19incyall.csv"),
        "n2_col":    4 
    }
}

POVERTY_COUNT_COLS = [
    "all_ages__poverty_estimate_all_ages",
    "all_0_17__poverty_estimate_age_0_17",
    "age_5_17_in_families__poverty_estimate_age_5_17_in_familes"
]

class WideScalarDataset: 

    def __init__(
        self,
        *,
        year: int, 
        saipe_csv: str,
        ir_csv: str,
        census_dir: str | Path
    ): 
        if year not in YEAR_SPECS:
            raise ValueError(f"unsupported year={year}, expected {sorted(YEAR_SPECS)}")

        self.year      = year 
        self.saipe_csv = Path(saipe_csv)
        self.ir_csv    = Path(ir_csv)

        if year == 2013: 
            self.label_map, _ = build_label_map(2013, census_dir=census_dir)
        else: 
            _, edges          = build_label_map(2013, census_dir=census_dir)
            self.label_map, _ = build_label_map(
                year, 
                train_edges=edges, 
                census_dir=census_dir
            )

    def build(self, *, n2_col: int) -> dict: 
        povc = self.load_poverty_counts()
        n2   = self.load_n2_returns(n2_col=n2_col)

        df = povc.merge(n2, on="fips", how="inner")
        df = df[df["fips"].isin(self.label_map.keys())].copy() 
        df["label"] = df["fips"].map(self.label_map)
        df = df.dropna(
            subset=[*POVERTY_COUNT_COLS, "n2_returns", "label"]
        ).reset_index(drop=True)

        feature_cols = ["n2_returns", *POVERTY_COUNT_COLS]
        X     = df[feature_cols].to_numpy(np.float64)
        y     = df["label"].to_numpy(np.float64).reshape(-1, 1)
        fips  = df["fips"].to_numpy(dtype="U5")
        names = np.array(feature_cols, dtype="U64")

        return {
            "features": X,
            "labels": y, 
            "fips_codes": fips, 
            "feature_names": names, 
        }

    def load_poverty_counts(self) -> pd.DataFrame: 

        raw = pd.read_csv(self.saipe_csv, header=None, dtype=str)
        if raw.shape[0] < 5:
            raise ValueError(f"invalid SAIPE csv format: {self.saipe_csv}")

        row_group = raw.iloc[2].tolist()
        row_sub   = raw.iloc[3].tolist()

        groups = []
        last = ""
        for g in row_group:
            if isinstance(g, str) and g.strip():
                last = g.strip()
            groups.append(last)

        name_to_idx = {}
        for i, sub in enumerate(row_sub):
            if i < 4:
                continue
            group_slug = (
                str(groups[i]).strip().lower()
                .replace(",", "")
                .replace("/", "_")
                .replace("-", "_")
            )
            group_slug = "_".join(group_slug.split())

            sub_raw = str(sub).strip()
            sub_low = sub_raw.lower()
            if sub_low.startswith("poverty estimate"):
                metric = (
                    sub_low
                    .replace(",", "")
                    .replace("/", "_")
                    .replace("-", "_")
                )
                metric = "_".join(metric.split())
                name = f"{group_slug}__{metric}"
                name_to_idx[name] = i

        required_source = {
            "all_ages__poverty_estimate_all_ages": "all_ages__poverty_estimate_all_ages",
            "all_0_17__poverty_estimate_age_0_17": "age_0_17__poverty_estimate_age_0_17",
            "age_5_17_in_families__poverty_estimate_age_5_17_in_familes":
                "age_5_17_in_families__poverty_estimate_age_5_17_in_families",
        }

        missing = [src for src in required_source.values() if src not in name_to_idx]
        if missing:
            raise ValueError(f"missing poverty estimate columns in {self.saipe_csv}: {missing}")

        data = raw.iloc[4:].reset_index(drop=True)
        state = data.iloc[:, 0].astype(str).str.strip().str.zfill(2)
        county = data.iloc[:, 1].astype(str).str.strip().str.zfill(3)
        fips = (state + county).astype(str)

        out = pd.DataFrame({"fips": fips.to_numpy()})
        for out_name, src_name in required_source.items():
            out[out_name] = to_num(data.iloc[:, name_to_idx[src_name]])

        out = out[(out["fips"] != "00000") & (~out["fips"].str.endswith("000"))].copy()
        out = out.groupby("fips", as_index=False)[POVERTY_COUNT_COLS].sum()
        out[POVERTY_COUNT_COLS] = out[POVERTY_COUNT_COLS].replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=POVERTY_COUNT_COLS).reset_index(drop=True)
        if not isinstance(out, pd.DataFrame): 
            raise RuntimeError 
        return out

    def load_n2_returns(self, *, n2_col: int) -> pd.DataFrame: 
        df = pd.read_csv(self.ir_csv, header=None, dtype=str, encoding="latin-1")
        df = df.iloc[6:].reset_index(drop=True)

        state_raw  = df.iloc[:, 0].astype(str).str.strip() 
        county_raw = df.iloc[:, 2].astype(str).str.strip() 
        valid      = state_raw.str.fullmatch(r"\d{1,2}") & county_raw.str.fullmatch(r"\d{1,3}")
        df         = df.loc[valid].reset_index(drop=True)

        state  = df.iloc[:, 0].astype(str).str.zfill(2)
        county = df.iloc[:, 2].astype(str).str.zfill(3)
        fips   = (state + county).astype(str)
        keep   = (~fips.str.endswith("000"))

        df     = df.loc[keep].reset_index(drop=True)
        fips   = fips[keep].reset_index(drop=True)

        n2  = to_num(df.iloc[:, n2_col])
        out = pd.DataFrame({"fips": fips.to_numpy(), "n2_returns": n2.to_numpy(np.float64)})
        out = out.groupby("fips", as_index=False)["n2_returns"].sum() 
        out = out[out["n2_returns"] > 0.0].copy() 
        if not isinstance(out, pd.DataFrame): 
            raise RuntimeError 
        return out 


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--year", type=int, required=True, choices=[2013, 2019])
    parser.add_argument("--out-mat", required=True)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    spec      = YEAR_SPECS[args.year]
    saipe_csv = spec["saipe_csv"]
    ir_csv    = spec["ir_csv"]
    n2_col    = spec["n2_col"]

    ds = WideScalarDataset(
        year=args.year,
        saipe_csv=saipe_csv,
        ir_csv=ir_csv,
        census_dir=project_path("data", "census")
    )
    data = ds.build(n2_col=n2_col)

    out = Path(args.out_mat)
    savemat(out, data) 
    print(f"[wide] saved {out} ({data['features'].shape[0]} rows, "
          f"{data['features'].shape[1]} feats)")

    if args.out_csv:
        df = pd.DataFrame(data["features"], columns=[str(c) for c in data["feature_names"]])
        df.insert(0, "fips", data["fips_codes"].astype(str))
        df["label"] = data["labels"].reshape(-1)
        csv_out = Path(args.out_csv)
        df.to_csv(csv_out, index=False)
        print(f"[wide] saved {csv_out}")


if __name__ == "__main__": 
    main() 
