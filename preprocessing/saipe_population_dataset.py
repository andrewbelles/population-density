#!/usr/bin/env python3 
# 
# saipe_population_dataset.py  Andrew Belles  Dec 21st, 2025 
# 
# Parses saipe json for years 2000, 2010, and 2020 matching up
# with population density for each county. 
# 
# Saipe data is mostly metrics on poverty/median income I think are 
# beneficial views in determining population density 
# 

import os, json
from typing import Dict 
import numpy as np 
import pandas as pd 

from scipy.io import savemat 
from support.helpers import project_path
from climate_population_dataset import PopulationDensity

VARS = [
    "SAEPOVRTALL_PT", 
    "SAEPOVALL_PT", 
    "SAEMHI_PT", 
    "SAEPOV0_17_PT", 
    "SAEPOVRTALL_MOE",
    "SAEMHI_MOE"
]


class SaipeDataset: 
    '''
    Parses SAIPE county JSON files and aligns years on canonical FIPS order. 
    '''

    def __init__(
        self, 
        saipe_dir: str | None = None, 
        census_dir: str | None = None, 
        geography_dir: str | None = None, 
        target_years: list[int] = [2000, 2010, 2020], 
        variables: list[str] = VARS
    ): 
        self.target_years = sorted(target_years)
        self.variables    = list(variables)

        if saipe_dir is None: 
            saipe_dir = project_path("data", "socioeconomic", "saipe", "raw")
        if census_dir is None:
            census_dir = project_path("data", "census")
        if geography_dir is None:
            geography_dir = project_path("data", "geography")

        self.saipe_dir = saipe_dir 
        self.population_data = PopulationDensity(
            census_dir, 
            geography_dir, 
            self.target_years
        )

        self.year_dataframes: Dict[int, pd.DataFrame] = {}
        self._process_years()
        self.features = self._align_on_fips() 

    def _process_years(self): 

        for year in self.target_years: 
            df = self._load_year(year)
            self.year_dataframes[year] = df 
            print(f"> Year {year}: {len(df)} counties processed")

    def _load_year(self, year: int) -> pd.DataFrame: 

        filepath = os.path.join(self.saipe_dir, f"saipe_{year}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"missing Saipe file: {filepath}")
        
        with open(filepath, 'r') as f: 
            data = json.load(f)

        if not isinstance(data, list) or len(data) < 2: 
            raise ValueError(f"invalid SAIPE JSON format: {filepath}")

        headers, rows = data[0], data[1:]
        df = pd.DataFrame(rows, columns=headers)

        required = {"NAME", "time", "state", "county", *self.variables}
        missing  = required - set(df.columns)
        if missing: 
            raise ValueError(f"missing required column(s): {sorted(missing)}")

        df["state"]  = df["state"].astype(str).str.zfill(2)
        df["county"] = df["county"].astype(str).str.zfill(3)
        df["FIPS"]   = df["state"] + df["county"]
        df["time"]   = pd.to_numeric(df["time"], errors="coerce")
        df = df[df["time"] == year].copy() 

        if not isinstance(df, pd.DataFrame): 
            raise TypeError 

        for col in self.variables: 
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["FIPS"])
        df = df.drop_duplicates(subset=["FIPS"]).sort_values("FIPS").reset_index(drop=True)

        res = df[["FIPS", "NAME", *self.variables]]
        if not isinstance(res, pd.DataFrame): 
            raise TypeError 
        return res 

    def _align_on_fips(self) -> Dict[int, pd.DataFrame]: 

        aligned = {}

        for year in self.target_years: 
            df = self.year_dataframes[year].copy() 
            df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
            
            label_col = f"density_{year}"
            if label_col not in self.population_data.df.columns: 
                raise ValueError(f"population density missing for year {year}")

            pop_subset = (self.population_data
                .df[["FIPS", label_col, "INTPTLAT", "INTPTLONG"]].copy()
            ) 
            if not isinstance(pop_subset, pd.DataFrame):
                raise TypeError 

            pop_subset = pop_subset.dropna(subset=[label_col])
            
            merged = df.merge(pop_subset, on="FIPS", how="inner")
            count  = len(merged)
            merged = merged.dropna() 
            dropped = count - len(merged)
            if dropped: 
                print(f"> Year {year}: dropped {dropped} rows with NaN's")
            aligned[year] = merged

        return aligned

    def _canonical_fips_order(self) -> list[str]: 

        fips_sets = []
        for year in self.target_years: 
            df = self.features.get(year)
            if df is None: 
                raise ValueError(f"missing dataframe for year {year}")
            fips = df["FIPS"].astype(str).str.zfill(5)
            fips_sets.append(set(fips.tolist()))

        common = set.intersection(*fips_sets) if fips_sets else set() 
        if not common: 
            raise ValueError("No common FIPS accross target years")
        return sorted(common)

    def save(self, output_path: str): 

        if not self.features: 
            raise ValueError("No features available to save")

        fips_order = np.asarray(self._canonical_fips_order(), dtype="U5") 

        year_data  = {}
        names_ref  = None 
        coords_ref = None 

        for year in self.target_years: 
            df = self.features.get(year) 
            if df is None: 
                raise TypeError 

            df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
            df = df.set_index("FIPS").loc[fips_order].reset_index()

            names = df["NAME"].astype(str).to_numpy() 
            if names_ref is None: 
                names_ref = names 
            elif not np.array_equal(names, names_ref): 
                raise ValueError(f"county NAME mismatch across years at {year}")
            
            coords = df[["INTPTLAT", "INTPTLONG"]].to_numpy(dtype=np.float64)
            if coords_ref is None: 
                coords_ref = coords 
            elif (coords.shape != coords_ref.shape or 
                  not np.allclose(coords, coords_ref, equal_nan=True)):
                raise ValueError(f"coords mismatch across years at {year}")

            label_col = f"density_{year}"
            if label_col not in df.columns: 
                raise ValueError(f"missing label column {label_col}")

            exclude_cols = ["FIPS", "NAME", "INTPTLAT", "INTPTLONG", label_col]
            feature_cols = [c for c in df.columns if c not in exclude_cols]

            features = df[feature_cols].to_numpy(dtype=np.float64)
            labels   = df[label_col].to_numpy(dtype=np.float64).reshape(-1, 1) 

            year_data[f"year_{year}"] = {
                "features": features,
                "labels": labels, 
                "feature_names": feature_cols, 
                "n_counties": len(df), 
                "fips_codes": fips_order,
                "coords": coords 
            }

        mat_data = {
            "fips_codes": fips_order, 
            "county_names": names_ref,
            "coords": coords_ref, 
            "n_counties": len(fips_order),
            "target_years": self.target_years, 
            "variables": self.variables, 
            "years": year_data 
        }

        savemat(output_path, mat_data)
        print(f"Saved .mat file: {output_path} for {len(year_data)} years")


def main(): 
    output_path = project_path("data", "datasets", "saipe_population.mat")
    dataset = SaipeDataset()
    dataset.save(output_path)

if __name__ == "__main__": 
    main() 

