#!/usr/bin/env python3 
# 
# cross_modal_dataset.py  Andrew Belles  Dec 31st, 2025 
# 
# Computes Cross-Modal Features for use in passthrough features 
#
# 

import argparse 

import numpy as np 
from numpy.typing import NDArray
import pandas as pd 
from scipy.io import loadmat, savemat 

from support.helpers import project_path, _mat_str_vector

class CrossModalDataset:

    def __init__(
        self,
        viirs_path: str | None = None, 
        tiger_path: str | None = None, 
        nlcd_path: str | None = None 
    ): 

        if viirs_path is None: 
            viirs_path = project_path("data", "datasets", "viirs_nchs_2023.mat")
        if tiger_path is None: 
            tiger_path = project_path("data", "datasets", "tiger_nchs_2023.mat")
        if nlcd_path is None: 
            nlcd_path = project_path("data", "datasets", "nlcd_nchs_2023.mat")

        self.viirs_path = viirs_path
        self.tiger_path = tiger_path
        self.nlcd_path  = nlcd_path

        self.df = self._build() 

    def _load_mat(self, path: str): 
        mat = loadmat(path)
        if "features" not in mat or "labels" not in mat or "fips_codes" not in mat: 
            raise ValueError(f"{path} missing required keys")

        X = np.asarray(mat["features"], dtype=np.float64)
        y = np.asarray(mat["labels"], dtype=np.int64).reshape(-1)
        fips = _mat_str_vector(mat["fips_codes"]).astype("U5")
        if "feature_names" not in mat: 
            raise ValueError(f"{path} missing feature_names")

        names = _mat_str_vector(mat["feature_names"]).astype("U64")
        names = np.array([n.strip() for n in names], dtype="U64")
        return X, y, fips, names 

    '''
    @staticmethod 
    def _zscore(x: NDArray) -> NDArray: 
        x = np.asarray(x, dtype=float)
    '''

    @staticmethod 
    def _pick(names: NDArray, X: NDArray, candidates: list[str]) -> NDArray:
        for name in candidates: 
            idx = np.where(names == name)[0]
            if idx.size > 0:
                return X[:, int(idx[0])]
        raise ValueError(f"missing feature names, tried: {candidates}")

    def _align(self, fips_a, fips_b, fips_c): 
        idx_a = {f: i for i, f in enumerate(fips_a)}
        idx_b = {f: i for i, f in enumerate(fips_b)}
        idx_c = {f: i for i, f in enumerate(fips_c)}

        common = [f for f in fips_a if f in idx_b and f in idx_c]
        if not common: 
            raise ValueError("no common FIPS across VIIRS/TIGER/NLCD")

        ia = [idx_a[f] for f in common]
        ib = [idx_b[f] for f in common]
        ic = [idx_c[f] for f in common]
        return np.array(common, dtype="U5"), ia, ib, ic 

    def _build(self) -> pd.DataFrame: 
        Xv, yv, fv, nv = self._load_mat(self.viirs_path)
        Xt, yt, ft, nt = self._load_mat(self.tiger_path)
        Xn, yn, fn, nn = self._load_mat(self.nlcd_path)

        fips, ia, ib, ic = self._align(fv, ft, fn)
        Xv, yv = Xv[ia], yv[ia]
        Xt, yt = Xt[ib], yt[ib]
        Xn, yn = Xn[ic], yn[ic]

        if not np.array_equal(yv, yt) or not np.array_equal(yv, yn):
            raise ValueError("label mismatch")

        viirs_mean = self._pick(nv, Xv, ["viirs_mean"])
        viirs_vrei = self._pick(nv, Xv, ["viirs_vrei"])
        viirs_grad = self._pick(nv, Xv, ["viirs_grad_mag"])

        tiger_integ = self._pick(nt, Xt, ["tiger_integration_r3"])
        tiger_mesh  = self._pick(nt, Xt, ["tiger_meshedness"])
        tiger_hwy   = self._pick(nt, Xt, ["tiger_density_hwy"])

        nlcd_nature = self._pick(nn, Xn, ["nlcd_nature"])

        radiance_entropy = viirs_vrei 
        dev_grad         = viirs_grad 
        vanui_proxy      = np.log1p(viirs_mean) * (1.0 - nlcd_nature)
        ems_proxy        = tiger_mesh 
        rez              = tiger_hwy * nlcd_nature 

        rows = pd.DataFrame({
            "FIPS": fips, 
            "label": yv, 
            "cross_viirs_log_mean": np.log1p(viirs_mean), 
            "cross_tiger_integ": tiger_integ,
            "cross_radiance_entropy": radiance_entropy,
            "cross_dev_intensity_gradient": dev_grad,
            "cross_vanui_proxy": vanui_proxy,
            "cross_effective_mesh_proxy": ems_proxy,
            "cross_road_effect_intensity": rez 
        })

        return rows.sort_values("FIPS").reset_index(drop=True)

    def save(self, output_path: str): 
        if self.df is None or self.df.empty: 
            raise ValueError("no dataset to save")
        feature_cols = [c for c in self.df.columns if c.startswith("cross_")]
        mat = {
            "features": self.df[feature_cols].to_numpy(dtype=np.float64),
            "labels": self.df["label"].to_numpy(dtype=np.int64).reshape(-1, 1),
            "feature_names": np.array(feature_cols, dtype="U"),
            "fips_codes": self.df["FIPS"].to_numpy(dtype="U5"),
        }
        savemat(output_path, mat)
        print(f"> Saved {output_path} ({self.df.shape[0]}) rows")


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", 
                        default=project_path("data", "datasets", "cross_modal_2023.mat"))
    args = parser.parse_args()

    dataset = CrossModalDataset()
    dataset.save(args.out)


if __name__ == "__main__": 
    main()
