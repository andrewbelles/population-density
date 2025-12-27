#!/usr/bin/env python3 
# 
# travel_times_dataset.py  Andrew Belles  Dec 26th, 2025 
# 
# Computes Dense (n_counties x n_counties) matrix representing 
# travel time from county a to b. Exports as .mat 
# 

import sys, os, argparse 
import pandas as pd
import numpy as np 

class TravelTime: 

    def __init__(
        self,
        input_path, 
        output_path, 
    ): 
        self.input_path  = input_path 
        self.output_path = output_path 
        input   = self._parse_txt() 
        self.df = self._compute_metrics(input) 
        self._save()

    def _parse_txt(self): 

        data = []

        try: 
            with open(self.input_path, 'r', encoding='latin-1') as f: 
                for line in f: 
                    stripped = line.strip() 
                    if not stripped or not stripped[0].isdigit(): 
                        continue 

                    parts = stripped.split() 

                    try: 
                        geoid_block = parts[0]
                        if len(geoid_block) != 12: 
                            continue 

                        dest_st   = int(geoid_block[0:3])
                        dest_cty  = geoid_block[3:6]
                        orig_st   = int(geoid_block[6:9])
                        orig_cty  = geoid_block[9:12]

                        dest_fips = f"{dest_st:02d}{dest_cty}"
                        orig_fips = f"{orig_st:02d}{orig_cty}"

                        raw_flow = parts[-2] 
                        if raw_flow == '.': 
                            continue 

                        flow = int(raw_flow.replace(',', ''))

                        if flow > 0: 
                            data.append((orig_fips, dest_fips, flow))

                    except (ValueError, IndexError):
                        continue 

        except FileNotFoundError: 
            print(f"input file not found at {self.input_path}")
            sys.exit(1)

        return pd.DataFrame(data, columns=["orig_fips", "dest_fips", "flow"])

    def _compute_metrics(self, df): 

        outflow_stats = df.groupby("orig_fips")["flow"].sum().reset_index() 
        outflow_stats.rename(columns={"flow": "total_outflow"}, inplace=True)

        df = df.merge(outflow_stats, on="orig_fips", how="left")

        df["probability"] = df["flow"] / (df["total_outflow"] + 1e-9)
        df["distance"] = -np.log(df["probability"] + 1e-9)

        return df[["orig_fips", "dest_fips", "distance", "flow", "probability"]]

    def _save(self): 

        if self.df is None or self.df.empty: 
            print("no data parsed. output file will be empty")
            return 

        prob_sums = self.df.groupby("orig_fips")["probability"].sum() 
        valid = prob_sums.between(1.0 - 1e-3, 1.0 + 1e-3).all() 
        if not valid: 
            raise ValueError("invalid probability sums")

        self.df.sort_values(by=["orig_fips", "distance"], inplace=True)
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        print(f"> Saved {len(self.df)} rows to {self.output_path}")


def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    TravelTime(args.input, args.output)


if __name__ == "__main__": 
    main() 
