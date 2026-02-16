#!/usr/bin/env python3 
# 
# labels.py  Andrew Belles  Feb 9th, 2026 
# 
# Label set of log population discretely binned in K (~8) discrete classes for ordinal regression 
# 
# 

import numpy as np 

import pandas as pd 

from typing        import Iterable 

from pathlib       import Path 

from dataclasses   import dataclass 

from utils.helpers import project_path 


@dataclass(frozen=True)
class BinConfig: 
    K: int | None = None 
    heuristic: str = "rice" 
    min_samples_per_bin: int = 40 
    min_k: int = 5 
    max_k: int = 32
    edge_margin: float = 0.05 

class PopulationLabels: 

    '''
    Label set of log-population binned into discrete K classes for ordinal regression via 
    log-uniform spacing. 
    ''' 

    # File to year mapping 
    FILE_BY_YEAR   = {
        2013: "co-est2020-alldata.csv", 
        2019: "co-est2020-alldata.csv",
    }

    # Columns to check for stable binning per year 
    AVAILABLE_COLS = {
        2013: ("POPESTIMATE2013",), 
        2019: ("POPESTIMATE2019",),
    } 

    def __init__(
        self,
        *,
        year: int, 
        census_dir: str | Path = project_path("data", "census"), 
        binning: BinConfig | None = None 
    ): 
        if year not in self.FILE_BY_YEAR: 
            raise ValueError(f"unsupported year={year}, expected {sorted(self.FILE_BY_YEAR)}")

        self.year       = year 
        self.census_dir = Path(census_dir)
        self.binning    = binning or BinConfig() 

        self.population_: pd.DataFrame | None = None 
        self.labels_: pd.DataFrame | None     = None 
        self.edges_: np.ndarray | None        = None 
        self.K_: int | None                   = None 

    def to_label_map(
        self,
        fips: Iterable[str] | None = None, 
        *,
        strict: bool = False, 
        one_based: bool = False 
    ) -> dict[str, int]: 
        if self.labels_ is None: 
            raise RuntimeError("call fit() before to_label_map()")

        df = self.labels_ 

        if fips is not None: 
            idx = pd.Index(self.normalize_fips(fips))
            df  = df.reindex(idx)

            missing = df["target_bin"].isna() 
            if strict and missing.any(): 
                raise KeyError(f"missing labels for FIPS")

            df = df[~missing]

        y = df["target_bin"].to_numpy(np.int64)
        if one_based: 
            y = y + 1 

        return {str(fid).zfill(5): int(lbl) for fid, lbl in zip(df.index, y)}

    def to_soft_rank_map(self, fips=None) -> dict[str, float]: 
        df = self.labels_ if fips is None else self.labels_.reindex(
            pd.Index(self.normalize_fips(fips))
        )
        
        if df is None: 
            raise TypeError 
        df = df[df["soft_rank"].notna()]
        return {
            str(fid).zfill(5): float(r) for fid, r in zip(df.index, df["soft_rank"].to_numpy())
        }

    def fit(
        self, 
        *, 
        feature_fips: Iterable[str] | None = None, 
        K: int | None = None
    ) -> "PopulationLabels": 
        population       = self.load_population_table()
        self.population_ = population  

        if feature_fips is None: 
            vals = population["log_pop"].to_numpy(np.float64)
        else: 
            idx  = pd.Index(self.normalize_fips(feature_fips))
            vals = population.reindex(idx)["log_pop"].dropna().to_numpy(np.float64)
            if vals.size == 0: 
                raise ValueError("feature_fips had no overlap with census population")

        K_eff = K if K is not None else int(self.binning.K or self.suggest_K(len(vals)))
        edges = self.fit_edges(vals, K_eff, edge_margin=self.binning.edge_margin)

        labels = population.copy() 
        labels["target_bin"] = np.searchsorted(
            edges[1:-1], 
            labels["log_pop"].to_numpy(np.float64), 
            side="right"
        ).astype(np.int64)

        labels["soft_rank"] = self.rank_from_edges(labels["log_pop"].to_numpy(np.float64), edges)

        self.labels_ = labels 
        self.edges_  = edges 
        self.K_      = int(edges.size - 1)
        return self 

    def load_population_table(self) -> pd.DataFrame: 
        df = pd.read_csv(self.source_path, encoding="latin-1", dtype={"STATE": str, "COUNTY": str})

        if not {"STATE", "COUNTY"}.issubset(df.columns): 
            raise KeyError("census file must contain STATE and COUNTY columns")

        col = self.resolve_col(df.columns)

        df["STATE"]  = df["STATE"].astype(str).str.zfill(2)
        df["COUNTY"] = df["COUNTY"].astype(str).str.zfill(3)

        if "SUMLEV" in df.columns: 
            df = df[df["SUMLEV"].astype(str).str.zfill(3) == "050"]
        else: 
            df = df[(df["STATE"] != "00") & (df["COUNTY"] != "000")]

        out = pd.DataFrame({
            "fips": (df["STATE"] + df["COUNTY"]).astype("string"),
            "pop":  pd.to_numeric(df[col], errors="coerce")
        })

        out = out.dropna(subset=["pop"])
        out = out[out["pop"] > 0].copy() 
        out = out.drop_duplicates(subset=["fips"])
        out["log_pop"] = np.log(out["pop"].to_numpy(np.float64))
        out = out.set_index("fips").sort_index()
        return out 

    @property 
    def source_path(self) -> Path: 
        p = self.census_dir / self.FILE_BY_YEAR[self.year]
        if not p.exists(): 
            raise FileNotFoundError(f"missing census file: {p}")
        return p 

    def resolve_col(self, columns: Iterable[str]) -> str: 
        cols = set(columns)
        for c in self.AVAILABLE_COLS[self.year]: 
            if c in cols: 
                return c 
        raise KeyError(
            f"no valid population column for year={self.year}"
            f"expected one of {self.AVAILABLE_COLS[self.year]}"
        )

    def suggest_K(self, n: int) -> int: 
        n = max(int(n), 2)

        if self.binning.heuristic == "sturges": 
            K = int(np.ceil(np.log2(n) + 1))
        elif self.binning.heuristic == "sqrt": 
            K = int(np.ceil(np.sqrt(n)))
        else: 
            K = int(np.ceil(2.0 * (n**(1.0 / 3.0))))

        K = max(self.binning.min_k, min(K, self.binning.max_k))
        K = min(K, max(2, n // max(1, self.binning.min_samples_per_bin)))
        return max(2, K)

    @staticmethod 
    def fit_edges(log_pop: np.ndarray, K: int, edge_margin: float = 0.0) -> np.ndarray: 
        vals = np.asarray(log_pop, dtype=np.float64).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2: 
            raise ValueError("fit_edges requires at least 2 finite samples")
        if K < 2: 
            raise ValueError(f"K must be >= 2, got {K}")

        lo   = float(vals.min())
        hi   = float(vals.max())
        span = max(hi - lo, 1e-9)

        if hi <= lo: 
            hi = lo + span 

        if edge_margin > 0.0: 
            lo -= edge_margin * span 
            hi += edge_margin * span 

        edges = np.linspace(lo, hi, K + 1, dtype=np.float64)
        if np.any(np.diff(edges) <= 0): 
            raise ValueError("degenerate log-uniform edges.")
        return edges

    @staticmethod
    def normalize_fips(values: Iterable[str]) -> list[str]: 
        return [str(v).strip().zfill(5) for v in values]

    @staticmethod 
    def rank_from_edges(log_pop: np.ndarray, edges: np.ndarray) -> np.ndarray: 
        K   = edges.size - 1
        idx = np.searchsorted(edges[1:-1], log_pop, side="right")
        idx = np.clip(idx, 0, K - 1)
        
        left  = edges[idx]
        right = edges[idx + 1]
        frac  = (log_pop - left) / np.maximum(right - left, 1e-9)
        frac  = np.clip(frac, 0.0, 1.0)

        max_rank = np.nextafter(float(K), 0.0)
        return np.clip(idx.astype(np.float64) + frac, 0.0, max_rank)

# ---------------------------------------------------------
# Leakage free convience function 
# ---------------------------------------------------------

def build_label_map(
    year: int,  
    *, 
    train_year: int = 2013, 
    census_dir: str | Path = project_path("data", "census"), 
    train_edges: np.ndarray | None = None
): 

    if year != train_year: 
        if train_edges is None: 
            train = PopulationLabels(year=train_year, census_dir=census_dir).fit()
            train_edges = train.edges_ 
            assert train_edges is not None 

        pop_target = PopulationLabels(year=year, census_dir=census_dir).load_population_table()
        y_target   = PopulationLabels.rank_from_edges(
            pop_target["log_pop"].to_numpy(np.float64), train_edges
        )

        label_map = {
            str(f).zfill(5): float(r)
            for f, r in zip(pop_target.index.to_numpy(), y_target)
        }
        return label_map, train_edges 
    
    fitted = PopulationLabels(year=year, census_dir=census_dir).fit()
    return fitted.to_soft_rank_map(), fitted.edges_ 
