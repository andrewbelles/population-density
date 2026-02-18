#!/usr/bin/env python3
#
# wide_scalar_dataset.py  Andrew Belles  Feb 10th
#
# Proxy scalar dataset for wide model.
# Uses poverty counts + poverty rates from SAIPE only.
#

import argparse
import re

from pathlib import Path

import pandas as pd
import numpy as np

from scipy.io import savemat

from utils.helpers import project_path, to_num
from preprocessing.labels import build_label_map


YEAR_SPECS = {
    2013: {
        "saipe_csv": project_path("data", "saipe", "saipe_2013.csv"),
    },
    2019: {
        "saipe_csv": project_path("data", "saipe", "saipe_2019.csv"),
    },
}

POVERTY_COUNT_COLS = [
    "all_ages__poverty_estimate_all_ages",
    "age_0_17__poverty_estimate_age_0_17",
    "age_5_17_in_families__poverty_estimate_age_5_17_in_families",
]

POVERTY_RATE_COLS = [
    "all_ages__poverty_percent_all_ages",
    "age_0_17__poverty_percent_age_0_17",
    "age_5_17_in_families__poverty_percent_age_5_17_in_families",
]

INCY_LAYOUT = {
    2013: {
        "file": "13incyall.csv",
        "total_returns": 4,
    },
    2019: {
        "file": "19incyall.csv",
        "total_returns": 4,
    },
}

LOG_N2_RETURNS_COL = "log_n2_returns"

PROXY_FEATURE_COLS = [*POVERTY_COUNT_COLS, LOG_N2_RETURNS_COL, *POVERTY_RATE_COLS]


class WideScalarDataset:

    def __init__(
        self,
        *,
        year: int,
        saipe_csv: str,
        census_dir: str | Path,
    ):
        if year not in YEAR_SPECS:
            raise ValueError(f"unsupported year={year}, expected {sorted(YEAR_SPECS)}")

        self.year = year
        self.saipe_csv = Path(saipe_csv)
        self.label_map = build_label_map(year, census_dir=census_dir)

    def build(self) -> dict:
        df = self.load_poverty_proxy()
        df = self.merge_log_n2_returns(df)
        df = df[df["fips"].isin(self.label_map.keys())].copy()
        df["label"] = df["fips"].map(self.label_map)
        df = df.dropna(subset=[*PROXY_FEATURE_COLS, "label"]).reset_index(drop=True)

        X = df[PROXY_FEATURE_COLS].to_numpy(np.float64)
        y = df["label"].to_numpy(np.float64).reshape(-1, 1)
        fips = df["fips"].to_numpy(dtype="U5")
        names = np.asarray(PROXY_FEATURE_COLS, dtype="U64")

        return {
            "features": X,
            "labels": y,
            "fips_codes": fips,
            "feature_names": names,
        }

    def load_poverty_proxy(self) -> pd.DataFrame:
        raw = pd.read_csv(self.saipe_csv, header=None, dtype=str)
        if raw.shape[0] < 5:
            raise ValueError(f"invalid SAIPE csv format: {self.saipe_csv}")

        row_group = raw.iloc[2].tolist()
        row_sub = raw.iloc[3].tolist()

        groups = []
        last = ""
        for g in row_group:
            if isinstance(g, str) and g.strip():
                last = g.strip()
            groups.append(last)

        def slug(text: str) -> str:
            s = str(text).strip().lower()
            s = re.sub(r"[^a-z0-9]+", "_", s)
            return s.strip("_")

        name_to_idx = {}
        for i, sub in enumerate(row_sub):
            if i < 4:
                continue
            group_slug = slug(groups[i])
            metric_slug = slug(sub)
            if metric_slug.startswith("poverty_estimate") or metric_slug.startswith("poverty_percent"):
                name_to_idx[f"{group_slug}__{metric_slug}"] = i

        required_source = {
            "all_ages__poverty_estimate_all_ages": "all_ages__poverty_estimate_all_ages",
            "age_0_17__poverty_estimate_age_0_17": "age_0_17__poverty_estimate_age_0_17",
            "age_5_17_in_families__poverty_estimate_age_5_17_in_families": (
                "age_5_17_in_families__poverty_estimate_age_5_17_in_families"
            ),
            "all_ages__poverty_percent_all_ages": "all_ages__poverty_percent_all_ages",
            "age_0_17__poverty_percent_age_0_17": "age_0_17__poverty_percent_age_0_17",
            "age_5_17_in_families__poverty_percent_age_5_17_in_families": (
                "age_5_17_in_families__poverty_percent_age_5_17_in_families"
            ),
        }

        missing = [src for src in required_source.values() if src not in name_to_idx]
        if missing:
            raise ValueError(f"missing poverty proxy columns in {self.saipe_csv}: {missing}")

        data = raw.iloc[4:].reset_index(drop=True)
        state = data.iloc[:, 0].astype(str).str.strip().str.zfill(2)
        county = data.iloc[:, 1].astype(str).str.strip().str.zfill(3)
        fips = (state + county).astype(str)

        out = pd.DataFrame({"fips": fips.to_numpy()})
        for out_name, src_name in required_source.items():
            out[out_name] = to_num(data.iloc[:, name_to_idx[src_name]])

        agg_map = {c: "sum" for c in POVERTY_COUNT_COLS}
        agg_map.update({c: "mean" for c in POVERTY_RATE_COLS})

        out = out[(out["fips"] != "00000") & (~out["fips"].str.endswith("000"))].copy()
        out = out.groupby("fips", as_index=False).agg(agg_map)

        # Counts follow a heavy-tailed distribution; compressing with log1p
        # improves conditioning for downstream HSIC/RBF distances.
        for c in POVERTY_COUNT_COLS:
            v = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            v = np.clip(v, a_min=0.0, a_max=None)
            out[c] = np.log1p(v)

        base_cols = [*POVERTY_COUNT_COLS, *POVERTY_RATE_COLS]
        out[base_cols] = out[base_cols].replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=base_cols).reset_index(drop=True)
        return out

    def merge_log_n2_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        spec = INCY_LAYOUT[self.year]
        incy_path = Path(self.saipe_csv).with_name(spec["file"])
        if not incy_path.exists():
            raise FileNotFoundError(f"missing IRS county file: {incy_path}")

        raw = pd.read_csv(incy_path, header=None, dtype=str, encoding="latin-1")
        raw = raw.iloc[6:].reset_index(drop=True)

        state_raw = raw.iloc[:, 0].astype(str).str.strip()
        county_raw = raw.iloc[:, 2].astype(str).str.strip()
        valid = state_raw.str.fullmatch(r"\d{1,2}") & county_raw.str.fullmatch(r"\d{1,3}")
        raw = raw.loc[valid].reset_index(drop=True)

        state = raw.iloc[:, 0].astype(str).str.zfill(2)
        county = raw.iloc[:, 2].astype(str).str.zfill(3)
        fips = (state + county).astype(str)

        keep = (fips != "00000") & (~fips.str.endswith("000"))
        raw = raw.loc[keep].reset_index(drop=True)
        fips = fips[keep].reset_index(drop=True)

        n2_returns = to_num(raw.iloc[:, spec["total_returns"]]).astype(np.float64)
        n2_df = pd.DataFrame({
            "fips": fips.to_numpy(dtype="U5"),
            "n2_returns": n2_returns.to_numpy(dtype=np.float64),
        })
        n2_df = n2_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["n2_returns"])
        n2_df = n2_df.groupby("fips", as_index=False)["n2_returns"].mean()

        vals = np.clip(n2_df["n2_returns"].to_numpy(dtype=np.float64, copy=False), a_min=0.0, a_max=None)
        n2_df[LOG_N2_RETURNS_COL] = np.log1p(vals)
        n2_df = n2_df[["fips", LOG_N2_RETURNS_COL]]

        out = df.merge(n2_df, on="fips", how="left")
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, choices=[2013, 2019])
    parser.add_argument("--out-mat", required=True)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    spec = YEAR_SPECS[args.year]
    ds = WideScalarDataset(
        year=args.year,
        saipe_csv=spec["saipe_csv"],
        census_dir=project_path("data", "census"),
    )
    data = ds.build()

    out = Path(args.out_mat)
    savemat(out, data)
    print(f"[wide] saved {out} ({data['features'].shape[0]} rows, {data['features'].shape[1]} feats)")

    if args.out_csv:
        df = pd.DataFrame(data["features"], columns=[str(c) for c in data["feature_names"]])
        df.insert(0, "fips", data["fips_codes"].astype(str))
        df["label"] = data["labels"].reshape(-1)
        csv_out = Path(args.out_csv)
        df.to_csv(csv_out, index=False)
        print(f"[wide] saved {csv_out}")


if __name__ == "__main__":
    main()
