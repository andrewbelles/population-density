#!/usr/bin/env python3 
# 
# cliamte_geospatial_dataset.py  Andrew Belles  Dec 15th, 2025 
# 
# Script to instantiate long term climate data labeled against 
# lat/lon for each county to determine if a "climate aware" 
# geospatial encoding is worth learning 
# 

import os, glob 
import pandas as pd 
import numpy as np 
import support.helpers as h 

from typing import Dict, List, Optional
from numpy.typing import NDArray
from scipy.io import savemat 

class ClimateGeospatial: 
    MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    
    def __init__(
        self, 
        climate_dir: str, 
        geography_dir: str, 
        variables: List[str], 
        groups: Dict[str, List[str]]
    ): 
    
        self.variables  = set(variables)
        self.groups     = groups 
        if not self._validate_groups_cover_variables(): 
            raise ValueError(f"{self.variables} must be partitioned on {groups}, groups isn't a disjoint partition")

        self.geo_df     = self._load_geometries(geography_dir)
        self.climate_df = self._load_climate(climate_dir) 

        if not isinstance(self.geo_df, pd.DataFrame):
            raise TypeError("geo_df failed type check at instantiation time")
        if not isinstance(self.climate_df, pd.DataFrame):
            raise TypeError("climate_df failed type check at instantiation time")

        self.dataset = self._clean_data(self.climate_df.merge(self.geo_df, on="FIPS", how="inner"))

    def _validate_groups_cover_variables(self) -> bool:
        
        # Union should equal original vars_set 
        union = set().union(*self.groups.values())
        if union != self.variables: 
            return False 

        # If equal then they are disjoint 
        total = sum(len(v) for v in self.groups.values())
        return total == len(union)

    def _load_geometries(
        self, 
        geography_dir: str
    ) -> pd.DataFrame: 
        
        gaz_path = os.path.join(geography_dir, "2020_Gaz_counties_national.txt")
        df = pd.read_csv(gaz_path, sep="\t", dtype={"GEOID": str})
        df.columns = df.columns.str.strip() 

        needed  = {"GEOID", "INTPTLAT", "INTPTLONG"}
        missing = needed - set(df.columns) 
        if missing: 
            raise ValueError(f"county gazetteer missing columns: {missing}")

        geo = df.rename(columns={"GEOID": "FIPS"})[["FIPS", "INTPTLAT", "INTPTLONG"]].copy()
        if not isinstance(geo, pd.DataFrame):
            raise TypeError("geo dataframe failed type check from copy")

        geo["FIPS"] = geo["FIPS"].astype(str).str.zfill(5)

        geo["INTPTLAT"]  = pd.to_numeric(geo["INTPTLAT"], errors="coerce")
        geo["INTPTLONG"] = pd.to_numeric(geo["INTPTLONG"], errors="coerce")

        geo = geo.dropna(subset=["INTPTLAT", "INTPTLONG"]).drop_duplicates(subset=["FIPS"])
        return geo.reset_index(drop=True)

    def _load_climate(
        self, 
        climate_dir: str
    ) -> Optional[pd.DataFrame]: 
    
        clim_files = self._get_climate_files(climate_dir)  

        tables: List[pd.DataFrame] = []
        for var, file in clim_files.items(): 
            var_data = self._load_single_climate_feature(file, var) 
            if var_data is None:
                continue 

            tables.append(var_data) 

        if not tables: 
            return None 

        features = tables[0]
        for df in tables[1:]:
            features = features.merge(df, on="FIPS", how="inner")
        return features

    def _get_climate_files(
        self, 
        dir: str
    ) -> Dict[str, str]:
        
        climate_files: Dict[str, str] = {}
        
        for var in sorted(self.variables): 
            files = sorted(glob.glob(os.path.join(dir, f"{var}_*.csv")))
            if not files: 
                raise FileNotFoundError(f"no parsed CSV found for {var} in {dir}")

            climate_files[var] = files[-1]

        return climate_files 

    def _load_single_climate_feature(
        self, 
        filepath: str, 
        var: str
    ) -> Optional[pd.DataFrame]: 
        
        df = pd.read_csv(filepath, dtype={"fips": str})
        df.columns = df.columns.str.strip().str.lower()

        needed  = {"fips", "year", *self.MONTHS}
        missing = needed - set(df.columns)
        if missing: 
            raise ValueError(f"{var}: missing columns in {os.path.basename(filepath)}: {missing}")

        out = df[["fips", "year", *self.MONTHS]].copy() 
        if not isinstance(out, pd.DataFrame): 
            raise TypeError("failed dataframe filter + copy")

        out = out.rename(columns={"fips": "FIPS"})
        out["FIPS"] = out["FIPS"].astype(str).str.zfill(5)
        out["year"] = pd.to_numeric(out["year"], errors="coerce")

        for m in self.MONTHS: 
            out[m] = pd.to_numeric(out[m], errors="coerce")
            out.loc[out[m].isin([-99.99, -99.9, -999.0, -9999.0]), m] = np.nan 

        agg = out.groupby("FIPS", as_index=False)[self.MONTHS].mean(numeric_only=True)
        if not isinstance(agg, pd.DataFrame): 
            raise TypeError("aggregation of monthly data failed type check")
        return agg.rename(columns={m: f"{var}_{m}" for m in self.MONTHS}) 

    def _clean_data(
        self, 
        df: pd.DataFrame
    ) -> Optional[pd.DataFrame]: 

        climate_cols = [f"{var}_{m}" for var in sorted(self.variables) for m in self.MONTHS]
        required     = {"FIPS", "INTPTLAT", "INTPTLONG", *climate_cols}
        missing      = required - set(df.columns)
        if missing: 
            raise ValueError(f"_clean_data missing required columns: {missing}")

        out = df.copy() 
        if not isinstance(out, pd.DataFrame): 
            raise TypeError("_clean_data copy failed type check")

        out["FIPS"] = out["FIPS"].astype(str).str.zfill(5)

        numeric_cols = ["INTPTLAT", "INTPTLONG", *climate_cols]
        for c in numeric_cols: 
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=["FIPS", *numeric_cols])
    
        out = out.drop_duplicates(subset=["FIPS"]).sort_values("FIPS").reset_index(drop=True)

        if out["FIPS"].duplicated().any(): 
            raise ValueError("_clean_data failed to enforce unique FIPS")

        return out 

    def save(
        self, 
        export_path: str
    ): 
        if self.dataset is None: 
            raise ValueError("dataset is None") 
        if not isinstance(self.dataset, pd.DataFrame): 
            raise TypeError("complete dataset failed type check")

        df = self.dataset.copy() 
        if not isinstance(df, pd.DataFrame): 
            raise TypeError("dataset copy failed type check")
        
        vars_sorted  = sorted(self.variables)
        climate_cols = [f"{v}_{m}" for v in vars_sorted for m in self.MONTHS]

        fips_codes  = df["FIPS"].to_numpy(dtype="U5")
        coords      = df[["INTPTLAT", "INTPTLONG"]].to_numpy(dtype=np.float64)
        climate_all = df[climate_cols].to_numpy(dtype=np.float64) 

        # Group division of feature data 

        group_blocks: Dict[str, NDArray[np.float64]] = {}
        group_feature_names: Dict[str, NDArray] = {}

        for name, vars in self.groups.items(): 
            cols = [f"{v}_{m}" for v in vars for m in self.MONTHS]
            group_blocks[name] = df[cols].to_numpy(dtype=np.float64)
            group_feature_names[name]  = np.array(cols, dtype="U64")

        

        mat: Dict[str, object] = {
            "fips_codes": fips_codes, 
            "features": climate_all, 
            "feature_names": np.array(climate_cols, dtype="U64"), 
            "labels": coords, 
            "label_names": np.array(["lat", "lon"], dtype="U8")
        }

        for name in self.groups.keys(): 
            mat[f"features_{name}"] = group_blocks[name]
            mat[f"feature_names_{name}"] = group_feature_names[name]

        savemat(export_path, mat) 
        print(f"> Saved climate geospatial dataset: {export_path}")
    
def main(): 

    '''
    Harness for Class to Create Dataset
    '''

    climate_dir   = h.project_path("data", "climate", "nclimdiv_county", "parsed")
    geography_dir = h.project_path("data", "geography") 
    
    variables = ["hddc", "cddc", "pdsi", "pmdi", "phdi", "zndx"] 
    groups    = {
        "degree_days": ["hddc", "cddc"],
        "palmer_indices": ["pdsi", "pmdi", "phdi", "zndx"]
    }

    dataset = ClimateGeospatial(climate_dir, geography_dir, variables, groups)
    dataset.save(h.project_path("data", "climate_geospatial.mat"))

if __name__ == "__main__": 
    main() 
