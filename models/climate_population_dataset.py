#!/usr/bin/env python 
# 
###########################################################
# 
# climate_population_dataset.py  Andrew Belles  Dec 13th, 2025 
# 
# Creates Climate View versus Population Density per county using  
# NOAA nclimdiv county dataset 
# 
# Incorporates the following fields into the view (on a per month basis): 
# tmpc: mean temperature 
# pcpn: precipitation 
# tmin: minimum temperature 
# tmax: maximum temperature 
# pdsi: palmer drought severity index, 
#   long-term moisture surplus/deficit  
# pmdi: palmer modified drought index, 
#   smoother transition b/t wet/dry spells 
# phdi: palmer hydrological drought index, 
#   represents hydrological impact of a drought 
# zndx: palmer z-index, short-term monthly moisture anomoly 
# hddc: heating degree days, 
#   measures how many days required external heating 
# cddc: cooling degree days, 
#   measures days requiring external cooling 
# 
########################################################### 

import os, glob, json 
import numpy as np 
import pandas as pd 

from scipy.io import savemat 
from sklearn.preprocessing import StandardScaler

from helpers import NCLIMDIV_RE, project_path
from typing import List, Dict, Optional 


class PopulationDensity: 
    '''
    Helper Class to Instantiate data structure containing population density information using 
    Census data from census.gov's NBER datasets 
    '''

    def __init__(self, census_dir: str, geography_dir: str, target_years: list): 
        '''
        Instantiates DataFrame. Requires call to load helpers for historical and 
        contemporary census datasets since census.gov adjusted format after 1990 

        Throws: 
            TypeErorr for failure to compute densities given census data and county metadata 

        Caller Provides: 
            census_dir, geography_dir as valid strings to datasets 
            target_years as years to extract from census data 
        '''
        self.target_years = sorted(target_years)
        self.meta         = PopulationDensity.load_county_metadata(geography_dir)

        census_data = {}
        census_data.update(self.load_historical_census_(census_dir))
        
        modern_years = []
        for year in target_years: 
            if year >= 2000: 
                modern_years.append(year)

        census_data.update(self.load_modern_census_(census_dir, modern_years))

        # census.gov gazetteer provides land mass in sqmi for us 
        # to compute population density as people per sqmi  
        
        self.df = self.compute_densities_for_years_(census_data)
        if not isinstance(self.df, pd.DataFrame): 
            raise TypeError("invalid dataframe for population density dataset")

        # Drops any rows with NaN's  
        self.df = self.df.dropna(
            subset=[col for col in self.df.columns if col.startswith("density_")]
        )


    def compute_densities_for_years_(self, census_data: dict): 
        '''
        Helper function to compute population densities as people per sqmi  
        
        Caller Provides: 
            census_data as a dict containing people per fips_code (county)

        We return: 
            population density indexed on fips_code per target year  
        '''


        result = self.meta.copy() 

        available_years = sorted(census_data.keys()) 

        for target_year in self.target_years: 
            pop_col = f"pop_{target_year}"
            density_col = f"density_{target_year}"


            print(f"> Processing {target_year}...")
            if target_year in available_years: 

                # Very gross but ensures we only handle data that has ALAND_SQMI and population 
                # for a single fips_code (county)

                result = result.merge(census_data[target_year], on="FIPS", how="left")
                mask = result[pop_col].notna() & (result["ALAND_SQMI"] > 0)
                result.loc[mask, density_col] = (
                        result.loc[mask, pop_col] / result.loc[mask, "ALAND_SQMI"]
                )
                result.loc[~mask, density_col] = np.nan

        return result 

    def load_historical_census_(self, census_dir: str): 
        '''
        Helper function to load any NBER census data from pre-2000

        Throws: 
            TypeError for failing to copy dataframe at specific year 

        Caller Provides: 
            census_dir as a valid string  

        We return: 
            people per county (indexed on fips code) for each target year < 2000 
        '''
        
        csv_path = os.path.join(census_dir, "county_population_1900_1990.csv")
        df = pd.read_csv(csv_path, dtype={"fips": str})

        df["fips"] = df["fips"].str.strip('"')
        df["fips"] = df["fips"].str.split(".").str[0]
        df["FIPS"] = df["fips"].str.zfill(5)

        # Pivot years into columns 
        census_years = {}
        for col in df.columns: 
            
            # This is a weird dance we have to do to ensure fips codes are interpreted from the csv 
            # and the coerced into a numeric value (which then aligns with how the modern NBER dataset works) 

            if col.startswith("pop") and col[3:].isdigit(): 
                year = int(col[3:])
                year_data = df[["FIPS", col]].copy() 
                if not isinstance(year_data, pd.DataFrame):
                    raise TypeError("copy error for year_data loading historical census data")
                year_data[col] = pd.to_numeric(year_data[col], errors="coerce")
                census_years[year] = year_data.rename(columns={col: f"pop_{year}"})

        return census_years 


    def load_modern_census_(self, census_dir: str, target_decades: List[int]):
        '''
        Helper function to load modern census data from NBER data >= 2000 

        Caller Provides: 
            census_dir as a valid string  

        We return: 
            people per county (indexed on fips code) for each target year < 2000 

        '''

        census_data = {}

        for decade in target_decades: 
            json_path = os.path.join(census_dir, f"county_population_{decade}.json")
            
            with open(json_path, 'r') as f: 
                data = json.load(f)

            headers, rows = data[0], data[1:]
            df = pd.DataFrame(rows, columns=headers)

            df["FIPS"] = df["state"] + df["county"]
            df[f"pop_{decade}"] = pd.to_numeric(df["P1_001N"])

            census_data[decade] = df[["FIPS", f"pop_{decade}"]]

        return census_data 

    @staticmethod 
    def load_county_metadata(dir: str): 
        '''
        Helper function to load county metadata from the gazatteer dataset. 
        I've made this a staticmethod since it's completely isolated from the Class itself. 

        Caller Provides: 
            dir (which is the geography_dir) as a valid_str 

        We return: 
            dict of metadata for each county in US (CONUS++)
        '''

        gaz_path = os.path.join(dir, "2020_Gaz_counties_national.txt")

        gaz = pd.read_csv(gaz_path, sep='\t', dtype={"GEOID": str})
        gaz = gaz.rename(columns={"GEOID": "FIPS"})
        gaz.columns = gaz.columns.str.strip()

        cols = ["FIPS", "USPS", "NAME", "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG"]
        available_cols = [col for col in cols if col in gaz.columns]
        return gaz[available_cols]

class ClimateDataset: 

    def __init__(
        self, 
        climate_dir: str | None = None, 
        census_dir: str | None = None,
        geography_dir: str | None = None, 
        target_decades: List[int] = [1990, 2000, 2010, 2020], 
        variables: List[str] = [
            "tmpc", "pcpn", "tmin", "tmax", "pdsi", "pmdi", 
            "phdi", "zndx", "hddc", "cddc"
        ]): 

        self.target_decades = target_decades
        self.variables      = variables

        if climate_dir is None: 
            climate_dir = project_path("data", "climate", "nclimdiv_county", "raw") 
        if census_dir is None: 
            census_dir = project_path("data", "census")
        if geography_dir is None: 
            geography_dir = project_path("data", "geography")

        self.climate_dir       = climate_dir 
        self.population_data   = PopulationDensity(census_dir, geography_dir, target_decades) 
        self.decade_dataframes = {}
        self._process_climate_data() 
        self.features = self._intersect()

    def _process_climate_data(self):

        climate_files = self._get_climate_files()
        
        for decade in self.target_decades: 
            decade_data = self._process_decade(decade, climate_files)
            if decade_data is not None: 
                self.decade_dataframes[decade] = decade_data 
                print(f"> Decade {decade}: {len(decade_data)} counties processed")
            else: 
                print(f"> Decade {decade} failed to find data")

    def _get_climate_files(self) -> Dict[str, str]: 

        climate_files = {}
        for file_path in glob.glob(os.path.join(self.climate_dir, "climdiv-*")):
            filename = os.path.basename(file_path)
            match = NCLIMDIV_RE.match(filename) 
            if match: 
                variable = match.group(1)
                if variable in self.variables: 
                    climate_files[variable] = file_path

        return climate_files

    def _process_decade(self, decade: int, 
                        climate_files: Dict[str, str]) -> Optional[pd.DataFrame]:

        decade_data = []

        for variable, file_path in climate_files.items(): 
            var_data = self._load_variable_data(file_path, variable, decade)
            if var_data is not None: 
                decade_data.append(var_data)

        if not decade_data: 
            return None 

        result = decade_data[0]
        for df in decade_data[1:]: 
            result = result.merge(df, on=["FIPS", "year", "month"], how="outer")

        return self._clean_decade_data(result)

    def _load_variable_data(self, file_path: str, variable: str, 
                            target_year: int) -> Optional[pd.DataFrame]: 
        try: 
            with open(file_path, 'r') as f:
                lines = f.readlines() 

            data = []
            for line in lines: 
                parts = line.strip().split() 
                if len(parts) >= 13: 
                    try:
                        code_str = parts[0]
                        if len(code_str) >= 11:
                            fips = code_str[:5]
                            year = int(code_str[7:11])

                            if year == target_year: 

                                for month_idx, value_str in enumerate(parts[1:13], 1): 
                                    try: 
                                        value = float(value_str) 
                                        data.append({
                                            "FIPS": fips, 
                                            "year": year, 
                                            "month": month_idx, 
                                            variable: value if value != -99.99 else np.nan 
                                        })
                                    except ValueError: 
                                        continue 
                                    
                    except (ValueError, IndexError): 
                        continue 

            return pd.DataFrame(data) if data else None
        except Exception as e: 
            print(f"[ERROR] {variable} data for year {target_year} failed: {e}")
            return None 

    def _clean_decade_data(self, df: pd.DataFrame) -> pd.DataFrame: 

        completeness = df.groupby("FIPS").agg({
            var: lambda x: x.notna().sum() for var in self.variables if var in df.columns 
        })

        total_months     = df.groupby("FIPS").size()
        good_counties    = completeness[
            (completeness == total_months.values.reshape(-1, 1)).all(axis=1)
        ].index 
        
        result = df[df["FIPS"].isin(good_counties)] 
        if not isinstance(result, pd.DataFrame):
            raise TypeError("result of clean/filter failed type check")

        return result  

    def summary(self, decade: int) -> pd.DataFrame: 

        if decade not in self.decade_dataframes: 
            raise ValueError(f"[D:{decade}] not processed by dataset")

        df = self.decade_dataframes[decade]

        result = df.pivot_table(
            index="FIPS", 
            columns=["month"],
            values=self.variables,
            aggfunc="first"
        ).reset_index() 

        result.columns = ["FIPS"] + [f"{var}_{month:02d}" for var, month in result.columns[1:]]
        return result 

    def _intersect(self) -> Dict[int, pd.DataFrame]: 
        decade_matrices = {}
        for decade in self.target_decades: 
            decade_data = self._intersect_decade(decade)
            decade_matrices[decade] = decade_data 
        return decade_matrices

    def _intersect_decade(self, decade: int) -> pd.DataFrame: 

        decade_summary = self.summary(decade)
        pop_col        = f"density_{decade}" 
        if pop_col not in self.population_data.df.columns: 
            raise ValueError(f"[ERROR] population data not available for {decade}")

        pop_subset = self.population_data.df[["FIPS", pop_col, "INTPTLAT", "INTPTLONG"]].copy() 
        if not isinstance(pop_subset, pd.DataFrame):
            raise TypeError(f"[ERROR] {decade}'s subset failed type check for pd.DataFrame")

        return decade_summary.merge(pop_subset, on="FIPS", how="inner")

    def save(self, output_path: str): 
        
        if not self.features: 
            raise ValueError("[ERROR] cannot save data as no data exists")

        first_decade = self.features[self.target_decades[0]]
        fips_codes   = first_decade["FIPS"].values 
        coords       = first_decade[["INTPTLAT", "INTPTLONG"]].values.astype(np.float64)

        decade_data = {} 
        for decade in self.target_decades: 
            decade_df = self.features[decade]

            exclude_cols = ["FIPS", "INTPTLAT", "INTPTLONG", "decade", f"density_{decade}"]
            feature_cols = [col for col in decade_df.columns if col not in exclude_cols]

            decade_features = decade_df[feature_cols].values.astype(np.float64) 
            decade_labels   = decade_df[f"density_{decade}"].values.astype(np.float64).reshape(-1, 1)

            decade_data[f"decade_{decade}"] = {
                "features": decade_features, 
                "labels": decade_labels, 
                "feature_names": feature_cols, 
                "n_counties": len(decade_df)
            }

            print(f"> Decade {decade}: {len(decade_df)} counties, {len(feature_cols)} features")

        mat_data = {
            "coords": coords, 
            "fips_codes": fips_codes, 
            "n_counties": len(fips_codes), 
            "target_decades": self.target_decades, 
            "variables": self.variables, 
            "decades": decade_data
        }

        savemat(output_path, mat_data)
        print(f"Saved .mat file: {output_path} for {len(decade_data)} decades")


def export_climate_county_metadata(decade_matrices: Dict[int, pd.DataFrame], output_path: str):
    '''
    Export the metadata for counties that remain within features_df to a tsv file 
    compatible with GeospatialGraph. 

    Caller Provides:
        features_df which is the complete features matrix from aggregated data. 
        output_path specifies output for tsv 
    '''
    first_decade = list(decade_matrices.values())[0]
    county_metadata = []

    for _, row in first_decade.iterrows(): 
        fips = row["FIPS"]

        metadata_row = {
            "USPS": row.get("USPS", "XX"),
            "GEOID": fips, 
            "ANSICODE": "00000000", 
            "NAME": row.get("NAME", ""),
            "ALAND": int(row.get("ALAND_SQMI", 1.0) * 2589988.11), 
            "AWATER": int(row.get("AWATER_SQMI", 0.0) * 2589988.11), 
            "ALAND_SQMI": row.get("ALAND_SQMI", 0.0), 
            "AWATER_SQMI": row.get("AWATER_SQMI", 0.0), 
            "INTPTLAT": row.get("INTPTLAT", 0.0), 
            "INTPTLONG": row.get("INTPTLONG", 0.0)
        }
        county_metadata.append(metadata_row)

    metadata_df = pd.DataFrame(county_metadata)
    metadata_df.to_csv(output_path, sep='\t', index=False)


def main(): 

    '''
    Compilation of Climate View from NOAA dataset and CENSUS data 
    '''

    dataset     = ClimateDataset() 
    output_path = project_path("data", "climate_population.mat")
    dataset.save(output_path)
    export_climate_county_metadata(dataset.features, project_path("data", "climate_counties.tsv"))

if __name__ == "__main__": 
    main() 
