#!/usr/env python 
# 
# climate_census.py  Andrew Belles  Dec 10th, 2025 
# 
# Aggregates Climate Data for use with Population density statistics 
# 
# 

import argparse, os, glob, json 
import numpy as np 
import pandas as pd 
import xarray as xr 
import geopandas as gpd 

from sklearn.preprocessing import StandardScaler
from rasterio.transform import from_bounds 
from rasterio.features import rasterize 
from scipy.io import savemat
 

class ClimateAgg: 
    '''
    Helper Class to Aggregate Climate Data by County on min, max, avg, and std for temp 
    as well as precipitation data per day for all years stored in the climate_dir  
    '''
    def __init__(self, geography_dir: str, climate_dir: str): 
        self.variables = ["tmax", "tmin", "tavg", "prcp"]
        
        counties_gdf = ClimateAgg.load_county_geometries(geography_dir)   

        self.cached_transform  = None 
        self.cached_grid_shape = None 

        all_results = []
        for year_dir in sorted(glob.glob(os.path.join(climate_dir, "[12]*"))):
            year = os.path.basename(year_dir)
            print(f"> Processing year {year}...")

            for nc_file in sorted(glob.glob(os.path.join(year_dir, "*.nc"))):
                filename = os.path.basename(nc_file)

                try: 
                    month = filename.split('-')[1][-2:]
                except: 
                    month = "unknown"
                
                print(f"    > Processing {filename}...")

                result = self.aggregate_single_file_(nc_file, counties_gdf)
                if result is not None: 
                    result["year"]        = int(year)
                    result["month"]       = int(month)
                    result["source_file"] = filename
                    all_results.append(result)
                    print(f"    > {filename} was successfully processed: {month}/{year}")

        if all_results: 
            self.df = pd.concat(all_results, ignore_index=True)
            self.df = self.clean_aggregated_dataframe_()
            if self.df is None: 
                raise RuntimeError("data failed to pass interpolate and filter")
        else: 
            self.df = None 


    def aggregate_single_file_(self, nc_path: str, counties_gdf): 
        ds = xr.open_dataset(nc_path) 

        if self.cached_transform is None: 
            if "lon" in ds.coords and "lat" in ds.coords: 
                lon = ds.coords["lon"].values 
                lat = ds.coords["lat"].values 
                self.cached_transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), 
                                                    len(lon), len(lat)) 
                self.cached_grid_shape = (len(lat), len(lon))
                print(f"        > Cached Transform for grid shape {self.cached_grid_shape}")

                self.compute_geometry_masks_(counties_gdf)

        all_vars = {}
        for var in self.variables: 
            if var not in ds.variables: 
                continue 

            # Remove extreneous dimensions from dataset such as time 
            data = ds[var]
            if len(data.dims) > 2: 
                data = data.isel(
                    {dim: 0 for dim in data.dims if dim not in ["lat", "lon", "x", "y"]})

            data = data.squeeze() 
            values = data.values 

            if values.ndim != 2: 
                print(f"        > Warning: {var} data is not 2D (shape: {values.shape}"
                       "skipping...")
                continue 

            all_vars[var] = values 

        ds.close() 

        if not all_vars: 
            return None 

        results = []
        for (fips, mask) in self.cached_masks: 
            county_row = {"FIPS": fips}

            for var, values in all_vars.items(): 
                if mask is not None and np.any(mask):
                      county_values = values[mask]
                      valid_values = (county_values[(county_values != -999) &
                                      ~np.isnan(county_values)])

                      if len(valid_values) > 0:
                          county_row[f"{var}_mean"] = np.mean(valid_values)
                          county_row[f"{var}_min"]  = np.min(valid_values)
                          county_row[f"{var}_max"]  = np.max(valid_values)
                          county_row[f"{var}_std"]  = np.std(valid_values)
                      else:
                          county_row[f"{var}_mean"] = np.nan
                          county_row[f"{var}_min"]  = np.nan
                          county_row[f"{var}_max"]  = np.nan
                          county_row[f"{var}_std"]  = np.nan
                else:
                    county_row[f"{var}_mean"] = np.nan
                    county_row[f"{var}_min"]  = np.nan
                    county_row[f"{var}_max"]  = np.nan
                    county_row[f"{var}_std"]  = np.nan

            results.append(county_row)
        return pd.DataFrame(results) 


    def compute_geometry_masks_(self, counties_gdf):
        self.cached_masks = []

        for _, county in counties_gdf.iterrows(): 
            fips = county["FIPS"]
            geometry = county["geometry"]

            try: 
                mask = rasterize(
                    [geometry], 
                    out_shape=self.cached_grid_shape,
                    transform=self.cached_transform, 
                    fill=0, 
                    default_value=1,
                    dtype="uint8",
                ).astype(bool)

                self.cached_masks.append((fips, mask))
            except Exception as e: 
                print(f"        > Warning: could not rasterize geometry for {fips}: {e}")
                self.cached_masks.append((fips, None))

        print(f"        > Cached {len(self.cached_masks)} geometry masks")
        

    def clean_aggregated_dataframe_(self):
        '''
        We need to filter counties that are not part of CONUS. As a safeguard, we also interpolate on counties that are missing 
        < 10% of their data (implying they belong to CONUS but are missing data)
        '''
        if self.df is None: 
            return None 

        agg = self.df.groupby("FIPS").agg({
            "tmax_mean": lambda x: x.notna().sum(), 
            "tmin_mean": lambda x: x.notna().sum(), 
            "FIPS": "count"
        })        
        
        if not isinstance(agg, pd.DataFrame):
            raise RuntimeError("failed to aggregate pivoted data")

        county_completeness = agg.rename(columns={"FIPS": "total_rows", "tmax_mean": "tmax_valid", "tmin_mean": "tmin_valid"})

        county_completeness["completion_rate"] = ((county_completeness["tmax_valid"] + county_completeness["tmin_valid"]) 
            / (2.0 * county_completeness["total_rows"]))

        good_counties = county_completeness[county_completeness["completion_rate"] > 0.9].index 

        df_filtered = self.df[self.df["FIPS"].isin(good_counties)].copy() 

        dropped_counties = len(county_completeness) - len(good_counties)
        dropped_rows     = len(self.df) - len(df_filtered)
        if dropped_counties > 0: 
            print(f"    > Dropped {dropped_counties} counties ({dropped_rows} rows) with >10% missing months")

        return self.interpolate_missing_(df_filtered)


    def interpolate_missing_(self, df: pd.DataFrame):
        '''
        Safely forward and backward fills continuous in time over data points that register as NaN 
        ''' 
        df_interp = df.copy() 
        
        feature_groups = {}
        for var in self.variables: 
            feature_groups[var] = [col for col in df.columns if col.startswith(f"{var}_")]

        for var_name, columns in feature_groups.items(): 
            if not columns: 
                continue 

            print(f"        > Interpolating {var_name}: {len(columns)} columns")

            var_data = df_interp[columns]

            sorted_cols = sorted(columns, key=ClimateAgg.safe_sort_key)
            var_data_sorted = var_data[sorted_cols]
            if not isinstance(var_data_sorted, pd.DataFrame): 
                raise RuntimeError(f"failed to get sorted data for current var {var_name}")

            # Try to use forward/backward fill temporally 
            var_data_filled = var_data_sorted.ffill(axis=1).bfill(axis=1)
            for col in sorted_cols: 
                if var_data_filled[col].isnull().any(): 
                    stat_type = col.split('_')[1]
                    similar_cols = [c for c in columns if f"_{stat_type}_" in c]

                    if len(similar_cols) > 1: 
                        similar_mean = df_interp[similar_cols].mean(axis=1)
                        var_data_filled[col].fillna(similar_mean, inplace=True)
                    else: 
                        var_data_filled[col].fillna(var_data_filled[col].mean(), inplace=True)
            
            df_interp[columns] = var_data_filled[columns]

        remaining_nans = df_interp.isnull().sum().sum() 
        if remaining_nans > 0: 
            print(f"    > Warning: {remaining_nans} NaN values remain, post-fill")
            df_interp.fillna(df_interp.mean(numeric_only=True), inplace=True) 
        
        return df_interp

    
    @staticmethod 
    def safe_sort_key(col_name: str): 
        parts = col_name.split('_')
        if len(parts) >= 4: 
            try: 
                year  = int(parts[2])
                month = int(parts[3])
                return (year, month)
            except (ValueError, IndexError): 
                return (0, 0)
        return (0, 0)

    @staticmethod 
    def load_county_geometries(dir: str): 
        '''
        Loads geometries of counties collected shapefile. 
        Also includes names of counties and their FIPS code. 
        
        Caller Provides: 
            dir: Path to geometry directory relative to ./models
        '''

        shapefile_path = os.path.join(dir, "county_shapefile", "tl_2020_us_county.shp")
        counties = gpd.read_file(shapefile_path)

        counties["FIPS"] = counties["STATEFP"] + counties["COUNTYFP"]
        counties = counties[["FIPS", "NAME", "STATEFP", "COUNTYFP", "geometry"]]
        return counties 


class PopulationDensity: 

    def __init__(self, census_dir: str, geography_dir: str, target_years: list): 
        self.target_years = sorted(target_years)
        self.meta         = PopulationDensity.load_county_metadata(geography_dir)

        census_data = {}
        census_data.update(self.load_historical_census_(census_dir))
        census_data.update(self.load_modern_census_(census_dir))

        self.df = self.compute_densities_for_years_(census_data)
        if not isinstance(self.df, pd.DataFrame): 
            raise RuntimeError("invalid dataframe for population density dataset")
        self.df = self.df.dropna(subset=[col for col in self.df.columns if col.startswith("density_")])


    def compute_densities_for_years_(self, census_data): 
        result = self.meta.copy() 

        available_years = sorted(census_data.keys()) 
        for target_year in self.target_years: 
            pop_col = f"pop_{target_year}"
            density_col = f"density_{target_year}"

            print(f"> Processing {target_year}...")
            if target_year in available_years: 

                print(f"    > Found direct data for {target_year}")
                result = result.merge(census_data[target_year], on="FIPS", how="left")
                mask = result[pop_col].notna() & (result["ALAND_SQMI"] > 0)
                result.loc[mask, density_col] = result.loc[mask, pop_col] / result.loc[mask, "ALAND_SQMI"]
                result.loc[~mask, density_col] = np.nan
                print(f"    > Merge completed, shape: {result.shape}")
            else:
                print(f"    > No direct data for {target_year}")

            density_col = f"density_{target_year}"
            pop_col     = f"pop_{target_year}"
            if pop_col in result.columns: 
                result[density_col] = result[pop_col] / result["ALAND_SQMI"]
                print(f"    > Valid density extracted")
            else: 
                result[density_col] = np.nan

        return result 

    def load_historical_census_(self, census_dir: str): 
        csv_path = os.path.join(census_dir, "county_population_1900_1990.csv")
        df = pd.read_csv(csv_path, dtype={"fips": str})

        df["fips"] = df["fips"].str.strip('"')
        df["fips"] = df["fips"].str.split(".").str[0]
        df["FIPS"] = df["fips"].str.zfill(5)

        # Pivot years into columns 
        census_years = {}
        for col in df.columns: 
            if col.startswith("pop") and col[3:].isdigit(): 
                year = int(col[3:])
                year_data = df[["FIPS", col]].copy() 
                if not isinstance(year_data, pd.DataFrame):
                    raise RuntimeError("copy error for year_data loading historical census data")

                year_data[col] = pd.to_numeric(year_data[col], errors="coerce")
                census_years[year] = year_data.rename(columns={col: f"pop_{year}"})
        
        return census_years 


    def load_modern_census_(self, census_dir: str):
        json_path = os.path.join(census_dir, "county_population_2020.json")
        
        with open(json_path, 'r') as f: 
            data = json.load(f)

        headers, rows = data[0], data[1:]
        df = pd.DataFrame(rows, columns=headers)

        df["FIPS"] = df["state"] + df["county"]
        df["pop_2020"] = pd.to_numeric(df["P1_001N"])

        return {2020: df[["FIPS", "pop_2020"]]}


    @staticmethod 
    def load_county_metadata(dir: str): 
        '''

        '''
        gaz_path = os.path.join(dir, "2020_Gaz_counties_national.txt")
        gaz = pd.read_csv(gaz_path, sep='\t', dtype={"GEOID": str})
        gaz = gaz.rename(columns={"GEOID": "FIPS"})
        cols = ["FIPS", "USPS", "NAME", "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG"]
        available_cols = [col for col in cols if col in gaz.columns]
        return gaz[available_cols]


def create_feature_matrix(climate_agg: ClimateAgg, pop_density: PopulationDensity):
    if climate_agg.df is None:
        raise ValueError("climate data not loaded and aggregated")

    if pop_density.df is None or not isinstance(pop_density.df, pd.DataFrame):
        raise ValueError("population density not computed and stored")

    # Collect common fips codes from each dataset 
    climate_fips = set(climate_agg.df["FIPS"].unique())
    pop_fips     = set(pop_density.df["FIPS"].unique())
    common_fips  = sorted(climate_fips.intersection(pop_fips))

    features = []
    for fips in common_fips: 
        county_data    = {"FIPS": fips}
        county_climate = climate_agg.df[climate_agg.df["FIPS"] == fips]
        
        for _, row in county_climate.iterrows(): 
            year  = int(row["year"])
            month = int(row["month"])

            for var in climate_agg.variables: 
                for stat in ["mean", "min", "max", "std"]:
                    col_name = f"{var}_{stat}"
                    if col_name in row: 
                        feature_name = f"{var}_{stat}_{year}_{month:02d}"
                        county_data[feature_name] = row[col_name]

        county_pop = pop_density.df[pop_density.df["FIPS"] == fips]
        if not isinstance(county_pop, pd.DataFrame):
            raise RuntimeError("failed to pull county populations for fips codes")

        if not county_pop.empty: 
            for target_year in pop_density.target_years:
                density_col = f"density_{target_year}"
                if density_col in county_pop.columns: 
                    county_data[density_col] = county_pop[density_col].iloc[0]

        features.append(county_data) 

    df = pd.DataFrame(features)
    df = df.set_index("FIPS")
    return df 


def save_df_as_mat(feature_df: pd.DataFrame, output_path: str): 

    fips_codes   = feature_df.index.values 

    if "INTPTLAT" in feature_df.columns and "INTPTLONG" in feature_df.columns:
        coords = feature_df[["INTPTLAT", "INTPTLONG"]].values.astype(np.float64)
    else:
        coords = np.full((len(fips_codes), 2), np.nan, dtype=np.float64)

    label_cols   = [col for col in feature_df.columns if col.startswith("density_")]
    decades      = sorted([int(col.split('_')[1]) for col in label_cols])
    feature_cols = [col for col in feature_df.columns if not col.startswith("density_")
                    and col not in ["INTPTLAT", "INTPTLONG"]] 

    features_per_decade = len(feature_cols) // len(decades)

    decade_data = {}
    for i, decade in enumerate(decades): 
        start_idx = i * features_per_decade 
        end_idx   = (i + 1) * features_per_decade if i < len(decades) - 1 else len(feature_cols)

        decade_feature_cols = feature_cols[start_idx:end_idx]
        decade_features     = feature_df[decade_feature_cols].values.astype(np.float64)
        decade_labels       = feature_df[f"density_{decade}"].values.astype(np.float64).reshape(-1, 1) 
         
        feature_scaler = StandardScaler()
        features = feature_scaler.fit_transform(decade_features)

        label_scaler = StandardScaler()
        labels   = label_scaler.fit_transform(decade_labels)

        decade_data[f"decade_{decade}"] = {
            "features": features,  
            "labels": labels, 
            "feature_names": decade_feature_cols, 
            "feature_mean": feature_scaler.mean_, 
            "feature_scale": feature_scaler.scale_, 
            "label_mean": label_scaler.mean_, 
            "label_scale": label_scaler.scale_ 
        }

    mat_data = {
        "coords": coords, 
        "fips_codes": fips_codes, 
        "n_counties": len(fips_codes), 
        "decades": decade_data
    }

    savemat(output_path, mat_data)
    print(f"Saved .mat file: {output_path} for {len(decades)} decades")


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--climdir", default="../data/climate")
    parser.add_argument("--censdir", default="../data/census")
    parser.add_argument("--geodir", default="../data/geography")
    args = parser.parse_args()

    population_density      = PopulationDensity(args.censdir, args.geodir, [1960, 1990, 2020]) 
    aggregated_climate_data = ClimateAgg(args.geodir, args.climdir)

    features = create_feature_matrix(aggregated_climate_data, population_density)
    if features.isnull().sum().sum() > 0: 
        print(f"Columns with NaN: {features.columns[features.isnull().any()].tolist()[:10]}")
    save_df_as_mat(features, "../data/climpop.mat")


if __name__ == "__main__":
    main() 
