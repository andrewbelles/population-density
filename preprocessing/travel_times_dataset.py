#!/usr/bin/env python3 
# 
# travel_times_dataset.py  Andrew Belles  Dec 26th, 2025 
# 
# Computes Dense (n_counties x n_counties) matrix representing 
# travel time from county a to b. Exports as .mat 
# 

import argparse 
import numpy as np 
import pandas as pd 

from preprocessing.loaders import (
    load_compact_dataset,
)

from models.graph_utils import (
    build_knn_graph_from_coords
)

from scipy.io import savemat

from support.helpers import project_path 

class TravelTime: 

    def __init__(
        self,
        tiger_path: str, 
        geo_path: str, 
        output_path: str, 
    ): 
        self.k = 50 
        self.tiger_path  = tiger_path 
        self.geo_path    = geo_path
        self.output_path = output_path 

        X_tiger, coords  = self._get_data()
        E, C, aff, sig   = self._compute_matrix(X_tiger, coords)

        mat = {
            "fips_codes": self.common, 
            "coords": coords, 
            "edge_index": E, 
            "edge_weight_cost": C, 
            "edge_weight_affinity": aff, 
            "sigma": sig 
        }

        savemat(self.output_path, mat)
        print(f"> Saved ({len(self.common)}x{len(self.common)}) matrix to {self.output_path}")

    def _get_data(self): 
        tiger_data  = load_compact_dataset(self.tiger_path)
        fips_tiger  = tiger_data["sample_ids"] 

        geo_df      = self._load_geo()
        fips_geo    = np.asarray(geo_df["FIPS"].values, dtype="U5")  
        common_fips = np.intersect1d(fips_tiger, fips_geo)
        idx_map_t   = {f: i for i, f in enumerate(fips_tiger)}
        indices_t   = [idx_map_t[f] for f in common_fips]

        X = tiger_data["features"][indices_t]
        C = geo_df.set_index("FIPS").loc[common_fips, ["INTPTLAT", "INTPTLONG"]] 
        C = np.asarray(C, dtype=np.float64)
        self.common = common_fips

        return X, C 

    def _load_geo(self): 
        df              = pd.read_csv(self.geo_path, sep="\t", dtype={"GEOID": str})
        df.columns      = df.columns.str.strip() 
        df["FIPS"]      = df["GEOID"].str.strip().str.zfill(5)
        df["INTPTLAT"]  = pd.to_numeric(df["INTPTLAT"], errors="coerce")
        df["INTPTLONG"] = pd.to_numeric(df["INTPTLONG"], errors="coerce")

        df     = df.dropna(subset=["INTPTLAT", "INTPTLONG"])
        return df 

    def _compute_matrix(self, X, C): 
        DENS_HWY, CIRCUITY = 0, 6

        circuity = X[:, CIRCUITY]
        hwy_dens = X[:, DENS_HWY]

        # For nan or 0 coerce into 1.0 implying roadways are a straight line 
        circuity[np.isnan(circuity) | (circuity < 1.0)] = 1.0

        graph = build_knn_graph_from_coords(C, k=self.k, directed=False)
        edge_index, dist_km = graph.to_coo_numpy() 
        src, dst = edge_index[0], edge_index[1]

        # Compute Cost from source to destination using the avg circuity b/t 
        avg_circuity = 0.5 * (circuity[src] + circuity[dst]) 
        proxy_cost   = dist_km * avg_circuity

        # Increase cost by how dense highways are in this county 
        avg_hwy      = 0.5 * (hwy_dens[src] + hwy_dens[dst])
        speed_factor = 1.0 + np.log1p(avg_hwy)
        proxy_cost   = proxy_cost / speed_factor 

        sigma = np.median(proxy_cost) 
        if sigma == 0: 
            sigma = 1.0 

        affinity = np.exp(-(proxy_cost**2) / (2.0 * sigma**2))

        return edge_index, proxy_cost, affinity, sigma 


def main(): 
    tiger_path_default = project_path("data", "datasets", "tiger_nchs_2023.mat")
    geo_path_default   = project_path("data", "geography", "2020_Gaz_counties_national.txt")
    out_path_default   = project_path("data", "datasets", "travel_proxy.mat")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tiger", default=tiger_path_default)
    parser.add_argument("--geo", default=geo_path_default)
    parser.add_argument("--out", default=out_path_default)
    args = parser.parse_args()

    TravelTime(args.tiger, args.geo, args.out)


if __name__ == "__main__": 
    main()
