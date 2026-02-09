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

from pathlib       import Path 

from scipy.io      import savemat 

from utils.helpers import project_path

from preprocessing.population_labels import build_label_map


class SaipeScalarDataset: 

    INCY_LAYOUT = {
        2013: {
            "file": "13incyall.csv",
            "total_returns": 4,
            "total_income_amount": 13,
            "wages_amount": 15,
            "dividends_amount": 19,
            "business_returns": 24,
            "unemployment_claims": 33,
            "social_security_returns": 35,
        },
        2019: {
            "file": "19incyall.csv",
            "total_returns": 4,
            "total_income_amount": 21,
            "wages_amount": 23,
            "dividends_amount": 27,
            "business_returns": 32,
            "unemployment_claims": 41,
            "social_security_returns": 43,
        },
    }

    RATE_FEATURES = [
        "wage_dependence_rate",
        "dividend_rate",
        "unemployment_rate",
        "gig_self_employed_rate",
        "retirement_rate"
    ]

    def __init__(
        self,
        csv_path: str | None = None, 
        label_year: int = 2013, 
        census_dir: str | Path = project_path("data", "census")
    ): 
        if csv_path is None: 
            csv_path = project_path("data", "saipe", "saipe_2013.csv")

        if label_year == 2013: 
            self.labels_map, _ = build_label_map(2013, census_dir=census_dir)
        else: 
            _, edges           = build_label_map(2013, census_dir=census_dir)
            self.labels_map, _ = build_label_map(label_year, train_edges=edges, census_dir=census_dir)
        self.label_year        = int(label_year)
        self.csv_path          = csv_path 
        self.data              = self._build()

    def save(self, out_path: str): 
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        savemat(out, self.data)

    def _build(self): 
        df   = pd.read_csv(self.csv_path, header=None, dtype=str)
        
        row_group = df.iloc[2].tolist() 
        row_sub   = df.iloc[3].tolist() 
        specs     = self.build_feature_spec(row_group, row_sub)

        specs     = [
            s for s in specs
            if s["group"] == "median_household_income"
            or s["name"] == "all_ages__poverty_percent_all_ages"
        ]

        col_idx = [s["idx"] for s in specs]
        feature_names = np.asarray([s["name"] for s in specs], dtype="U64")
        
        data   = df.iloc[4:].reset_index(drop=True)
        state  = data.iloc[:, 0].astype(str).str.zfill(2)
        county = data.iloc[:, 1].astype(str).str.zfill(3)
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
        fips      = fips[keep_mask].reset_index(drop=True)
        X         = X.loc[keep_mask].reset_index(drop=True)

        saipe_df = pd.DataFrame(X)
        saipe_df.columns = feature_names
        saipe_df.insert(0, "fips", fips.astype(str).to_numpy())

        incy_df = self._load_incy_rates()
        full_df = saipe_df.merge(incy_df, on="fips", how="left")
        full_df = full_df.dropna().reset_index(drop=True)

        fips = full_df["fips"].to_numpy(dtype="U5")
        feature_cols = [c for c in full_df.columns if c != "fips"]
        X = full_df[feature_cols].to_numpy(dtype=np.float64)
        y = np.array([self.labels_map[f] for f in fips], dtype=np.int64).reshape(-1, 1)
        feature_names = np.asarray(feature_cols, dtype="U64")

        return {
            "features": X,
            "labels": y,
            "fips_codes": fips,
            "feature_names": feature_names
        }

    def _load_incy_rates(self) -> pd.DataFrame:
        year = 2019 if self.label_year != 2013 else 2013
        spec = self.INCY_LAYOUT[year]

        incy_path = Path(self.csv_path).with_name(spec["file"])
        if not incy_path.exists():
            raise FileNotFoundError(f"missing IRS county file: {incy_path}")

        df = pd.read_csv(incy_path, header=None, dtype=str, encoding="latin-1")
        df = df.iloc[6:].reset_index(drop=True)

        state_raw = df.iloc[:, 0].astype(str).str.strip()
        county_raw = df.iloc[:, 2].astype(str).str.strip()
        valid = state_raw.str.fullmatch(r"\d{1,2}") & county_raw.str.fullmatch(r"\d{1,3}")
        df = df.loc[valid].reset_index(drop=True)

        state = df.iloc[:, 0].astype(str).str.zfill(2)
        county = df.iloc[:, 2].astype(str).str.zfill(3)
        fips = (state + county).astype(str)

        keep = (fips != "00000") & (~fips.str.endswith("000"))
        df = df.loc[keep].reset_index(drop=True)
        fips = fips[keep].reset_index(drop=True)

        total_returns = self._to_num(df.iloc[:, spec["total_returns"]])
        total_income = self._to_num(df.iloc[:, spec["total_income_amount"]])
        wages = self._to_num(df.iloc[:, spec["wages_amount"]])
        dividends = self._to_num(df.iloc[:, spec["dividends_amount"]])
        business_returns = self._to_num(df.iloc[:, spec["business_returns"]])
        unemployment_claims = self._to_num(df.iloc[:, spec["unemployment_claims"]])
        social_security_returns = self._to_num(df.iloc[:, spec["social_security_returns"]])

        income_denom = total_income.where(total_income > 0.0)
        return_denom = total_returns.where(total_returns > 0.0)

        out = pd.DataFrame({
            "fips": fips.to_numpy(),
            "wage_dependence_rate": wages / income_denom,
            "dividend_rate": dividends / income_denom,
            "unemployment_rate": unemployment_claims / return_denom,
            "gig_self_employed_rate": business_returns / return_denom,
            "retirement_rate": social_security_returns / return_denom
        })

        out[self.RATE_FEATURES] = out[self.RATE_FEATURES].replace([np.inf, -np.inf], np.nan)
        out = out.groupby("fips", as_index=False)[self.RATE_FEATURES].mean()
        return out

    @staticmethod
    def _to_num(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

    @staticmethod
    def build_feature_spec(row_group, row_sub):
        groups = []
        last   = None 
        for g in row_group:
            if isinstance(g, str) and g.strip(): 
                last = g.strip() 
            groups.append(last or "")

        specs = []
        for i, sub in enumerate(row_sub): 
            if i < 4: 
                continue 
            g_raw = str(groups[i]).strip() 
            s_raw = str(sub).strip()
            g     = SaipeScalarDataset._state_slug(g_raw)
            s     = s_raw.lower() 

            if s.startswith("poverty estimate"): 
                continue 

            if s == "median household income":
                metric = "value" 
            elif s == "90% ci lower bound": 
                metric = "ci90_lower"
            elif s == "90% ci upper bound": 
                metric = "ci90_upper"
            else: 
                metric = SaipeScalarDataset._state_slug(s_raw)

            specs.append({"idx": i, "group": g, "name": f"{g}__{metric}"})
        return specs 

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
        }.get(base, SaipeScalarDataset._state_slug(base))

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
            group     = SaipeScalarDataset._state_slug(str(group_raw))
            metric    = SaipeScalarDataset._metric_name(str(sub))
            name      = f"{group}__{metric}" if group != metric else metric 
            names.append(name)
            col_idx.append(i)

        return names, col_idx 


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=project_path("data", "saipe", "saipe_2019.csv"))
    parser.add_argument("--year", type=int, default=2013)
    parser.add_argument("--census-dir", default=project_path("data", "census"))
    parser.add_argument("--out", default=project_path("data", "datasets",
                                                      "saipe_scalar_2019.mat"))
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    data = SaipeScalarDataset(args.csv, args.year, args.census_dir)
    data.save(args.out)

    n = data.data["features"].shape[0]
    d = data.data["features"].shape[1]

    print(f"[saipe] saved {args.out} ({n} rows, {d} features)")

    if args.out_csv: 
        feature_cols = [str(c) for c in data.data["feature_names"].tolist()]
        out_df = pd.DataFrame(data.data["features"], columns=feature_cols)
        out_df.insert(0, "fips", data.data["fips_codes"].astype(str))
        out_df["label"] = data.data["labels"].reshape(-1).astype(int)
        out_df.to_csv(args.out_csv, index=False)
        print(f"[saipe] saved {args.out_csv}")

if __name__ == "__main__": 
    main() 
