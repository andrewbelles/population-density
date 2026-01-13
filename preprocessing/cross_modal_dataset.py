
import argparse 

import numpy as np 

from dataclasses import dataclass 

from numpy.typing import NDArray 

from scipy.io import loadmat, savemat 

from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestClassifier

from utils.helpers import (
    project_path,
    _mat_str_vector,
    align_on_fips 
)

from preprocessing.loaders import load_oof_predictions

EPS          = 1e-6 
RANDOM_STATE = 0 

@dataclass 
class MatDataset: 
    X:     NDArray[np.float64]
    y:     NDArray[np.int64]
    fips:  NDArray[np.str_]
    names: NDArray[np.str_]

    @staticmethod
    def load(path: str) -> "MatDataset": 
        mat = loadmat(path)
        required = ("features", "labels", "fips_codes", "feature_names")
        if any(k not in mat for k in required): 
            missing = [k for k in required if k not in mat]
            raise ValueError(f"{path} missing required keys: {missing}")

        X = np.asarray(mat["features"], dtype=np.float64)
        if X.ndim == 1: 
            X = X.reshape(-1, 1)

        y     = np.asarray(mat["labels"]).reshape(-1).astype(np.int64)
        fips  = _mat_str_vector(mat["fips_codes"]).astype("U5")
        names = _mat_str_vector(mat["feature_names"]).astype("U64")
        names = np.array([n.strip() for n in names], dtype="U64")

        if X.shape[1] != names.shape[0]: 
            names = np.array([f"f{i}" for i in range(X.shape[1])], dtype="U64")

        return MatDataset(X=X, y=y, fips=fips, names=names)

def pick_feature(
    names: NDArray[np.str_], 
    X: NDArray[np.float64], 
    candidates: list[str]
) -> NDArray: 

    for name in candidates: 
        idx = np.where(names == name)[0]
        if idx.size > 0: 
            return X[:, int(idx[0])]
    raise ValueError(f"missing feature name, tried: {candidates}")

def align_datasets(
    base: MatDataset, 
    *others: MatDataset
) -> tuple[NDArray[np.str_], list[MatDataset]]:

    other_sets = [set(d.fips.tolist()) for d in others]
    common     = [f for f in base.fips if all(f in s for s in other_sets)]
    if not common: 
        raise ValueError("no common FIPS across datasets")

    aligned = []
    for d in (base, *others): 
        idx = align_on_fips(common, d.fips)
        aligned.append(
            MatDataset(
                X=d.X[idx],
                y=d.y[idx],
                fips=np.array(common, dtype="U5"),
                names=d.names 
            )
        )

    return np.array(common, dtype="U5"), aligned 

def stack_raw_features(
    datasets: dict[str, MatDataset]
) -> tuple[NDArray[np.float64], NDArray[np.str_]]: 
    X_blocks    = []
    name_blocks = []
    for prefix, data in datasets.items(): 
        X_blocks.append(data.X)
        names = np.array([f"{prefix}__{n}" for n in data.names], dtype="U64")
        name_blocks.append(names)
    X_all     = np.hstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]
    names_all = np.concatenate(name_blocks) if len(name_blocks) > 1 else name_blocks[0]
    return X_all, names_all 

def drop_nan_rows(
    X: NDArray[np.float64], 
    y: NDArray[np.int64],
    fips: NDArray[np.str_]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.str_]]: 

    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    mask = np.isfinite(X).all(axis=1)
    return X[mask], y[mask], fips[mask]

def load_oof_preds(
    path: str, 
    *,
    model_name: str | None = None
) -> tuple[NDArray[np.int64], NDArray[np.str_]]: 
    oof   = load_oof_predictions(path)
    preds = np.asarray(oof["preds"])
    if preds.ndim == 2: 
        if preds.shape[1] == 1:
            preds = preds.reshape(-1)
        else: 
            model_names = [str(m) for m in oof["model_names"]]
            if model_name is None: 
                raise ValueError(f"{path} has multiple models, specify model_name")
            if model_name not in model_names: 
                raise ValueError(f"{path} model_name not found: {model_name}") 
            preds = preds[:, model_names.index(model_name)]
    else: 
        preds = preds.reshape(-1)
    fips = np.asarray(oof["fips_codes"]).astype("U5")
    return preds.astype(np.int64), fips 

def compute_disagreement(
    fips: NDArray[np.str_],
    oof_paths: dict[str, str],
    *,
    model_names: dict[str, str] | None = None 
) -> NDArray[np.int64]: 
    pred_blocks = []
    for name, path in oof_paths.items(): 
        preds, fips_oof = load_oof_preds(path, model_name=(model_names or {}).get(name))
        idx = align_on_fips(fips, fips_oof)
        pred_blocks.append(preds[idx])

    preds_all = np.stack(pred_blocks, axis=1)
    agreement = np.all(preds_all == preds_all[:, [0]], axis=1)
    return (~agreement).astype(np.int64)

def select_disagreement_features(
        X_raw: NDArray[np.float64],
        raw_names: NDArray[np.str_],
        y_conflict: NDArray[np.int64],
    *,
    n_features: int = 3, 
    random_state: int = 0 
) -> tuple[NDArray[np.float64], list[str]]: 
    
    mask = np.isfinite(X_raw).all(axis=1)
    X    = X_raw[mask]
    y    = y_conflict[mask]

    if X.shape[0] == 0: 
        raise ValueError("no valid rows for disagreement training")

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=random_state,
        n_jobs=-1, 
        class_weight="balanced_subsample"
    )

    rf.fit(X, y)

    importances = np.asarray(rf.feature_importances_, dtype=np.float64)
    if importances.size == 0: 
        raise ValueError("no feature importances computed")

    top_idx    = np.argsort(importances)[::-1][:n_features]
    top_idx    = [int(i) for i in top_idx]
    top_names  = [str(raw_names[i]) for i in top_idx]
    top_values = X_raw[:, top_idx]
    return top_values, top_names 

def residual_correlation(
    feature_names: list[str],
    X: NDArray[np.float64],
    fips: NDArray[np.str_],
    *,
    oof_path: str,
    model_name: str | None = None 
) -> dict[str, float]: 
    
    oof   = load_oof_predictions(oof_path)
    preds = np.asarray(oof["preds"])
    if preds.ndim == 2 and preds.shape[1] > 1: 
        model_names = [str(m) for m in oof["model_names"]]
        if model_name is None: 
            raise ValueError(f"{oof_path} has multiple models, specify model_name")
        if model_name not in model_names: 
            raise ValueError(f"{oof_path} model_name not found: {model_name}")
        preds = preds[:, model_names.index(model_name)]
    preds  = preds.reshape(-1)
    y_true = np.asarray(oof["labels"]).reshape(-1)

    fips_oof = np.asarray(oof["fips_codes"]).astype("U5")
    idx      = align_on_fips(fips, fips_oof)
    residual = y_true[idx] - preds[idx]

    correlations = {}
    for i, name in enumerate(feature_names): 
        x = X[:, i]
        if np.allclose(x, x[0]): 
            correlations[name] = 0.0 
            continue 
        r, _ = spearmanr(x, residual)
        correlations[name] = float(r) if np.isfinite(r) else 0.0 
    return correlations 

# --------------------------------------------------------- 
# Build Entry  
# --------------------------------------------------------- 

def build_cross_modal_features(
    *,
    viirs_path: str, 
    tiger_path: str, 
    nlcd_path: str,
    saipe_path: str, 
    oof_paths: dict[str, str],
    disagreement_features: int = 3, 
    random_state: int = 0 
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.str_], list[str], list[str]]: 

    viirs = MatDataset.load(viirs_path)
    tiger = MatDataset.load(tiger_path)
    nlcd  = MatDataset.load(nlcd_path)
    saipe = MatDataset.load(saipe_path)

    fips, (viirs, tiger, nlcd, saipe) = align_datasets(viirs, tiger, nlcd, saipe)

    if not (np.array_equal(viirs.y, tiger.y) and 
            np.array_equal(viirs.y, nlcd.y) and 
            np.array_equal(viirs.y, saipe.y)): 
        raise ValueError("label mismatch across datasets")

    # Compute cross modal features 
    viirs_mean      = pick_feature(viirs.names, viirs.X, ["viirs_mean"])
    income          = pick_feature(saipe.names, saipe.X, ["median_household_income"])
    log_viirs       = np.log1p(viirs_mean)
    log_income      = np.log1p(income)
    urban_wealth    = log_viirs * log_income 
    rural_wealth    = log_income / (log_viirs + EPS)
    tiger_ratio_4   = pick_feature(tiger.names, tiger.X, ["tiger_ratio_4way"])
    tiger_ratio_3   = pick_feature(tiger.names, tiger.X, ["tiger_ratio_3way"])
    grid_complexity = (tiger_ratio_4 + EPS) / (tiger_ratio_3 + EPS) 
    nlcd_diversity  = pick_feature(nlcd.names, nlcd.X, ["nlcd_diversity"])
    nlcd_contagion  = pick_feature(nlcd.names, nlcd.X, ["nlcd_contagion"])
    fragmented_dev  = (nlcd_diversity + EPS) / (nlcd_contagion + EPS)
    nlcd_dev_open   = pick_feature(nlcd.names, nlcd.X, ["nlcd_dev_open"])
    nlcd_edge_open  = pick_feature(nlcd.names, nlcd.X, ["nlcd_edge_dens_dev_open"])
    edge_openness   = nlcd_dev_open / (nlcd_edge_open + EPS)

    features = [
        urban_wealth,
        rural_wealth,
        grid_complexity,
        fragmented_dev,
        edge_openness
    ]

    feature_names = [
        "cross_urban_wealth_index",
        "cross_rural_wealth_ratio",
        "cross_grid_complexity_ratio", 
        "cross_fragmented_development_ratio",
        "cross_edge_openness_ratio"
    ]

    raw_blocks, raw_names = stack_raw_features({
        "viirs": viirs, 
        "tiger": tiger,
        "nlcd": nlcd, 
        "saipe": saipe 
    })
    y_conflict = compute_disagreement(fips, oof_paths)

    top_values, top_raw_names = select_disagreement_features(
        raw_blocks, 
        raw_names, 
        y_conflict, 
        n_features=disagreement_features,
        random_state=random_state
    )
    for i, raw in enumerate(top_raw_names): 
        features.append(top_values[:, i])
        feature_names.append(raw)

    X = np.column_stack(features).astype(np.float64)
    y = viirs.y.astype(np.int64)

    X, y, fips = drop_nan_rows(X, y, fips)
    return X, y, fips, feature_names, top_raw_names

# ---------------------------------------------------------
# Main Entry 
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--viirs", default=project_path("data", "datasets", 
                                                        "viirs_nchs_2023.mat"))
    parser.add_argument("--tiger", default=project_path("data", "datasets", 
                                                        "tiger_nchs_2023.mat"))
    parser.add_argument("--nlcd", default=project_path("data", "datasets", 
                                                        "nlcd_nchs_2023.mat"))
    parser.add_argument("--saipe", default=project_path("data", "datasets", 
                                                        "saipe_nchs_2023.mat"))
    parser.add_argument("--oof-nlcd", default=project_path("data", "stacking", 
                                                            "nlcd_optimized_probs.mat"))
    parser.add_argument("--oof-saipe", default=project_path("data", "stacking", 
                                                            "saipe_optimized_probs.mat"))
    parser.add_argument("--oof-viirs", default=project_path("data", "stacking", 
                                                            "viirs_optimized_probs.mat"))
    parser.add_argument("--out", default=project_path("data", "datasets", 
                                                      "cross_modal_2023.mat"))
    parser.add_argument("--residual-oof", default=project_path("data", "stacking",
                                                               "stacking_passthrough_oof.mat"))
    parser.add_argument("--residual-threshold", type=float, default=0.1)
    parser.add_argument("--filter-by-corr", action="store_true")
    args = parser.parse_args()

    oof_paths = {
        "viirs": args.oof_viirs, 
        "nlcd":  args.oof_nlcd, 
        "saipe": args.oof_saipe
    }

    X, y, fips, feature_names, top_raw_names = build_cross_modal_features(
        viirs_path=args.viirs,
        tiger_path=args.tiger,
        nlcd_path=args.nlcd,
        saipe_path=args.saipe,
        oof_paths=oof_paths,
        random_state=RANDOM_STATE
    )

    correlations = None 
    if args.residual_oof: 
        correlations = residual_correlation(
            feature_names, 
            X, 
            fips, 
            oof_path=args.residual_oof
        )
        print("> Residual Correlations (r):")
        for name, r in correlations.items(): 
            keep = "keep" if abs(r) > args.residual_threshold else "discard"
            print(f"    {name}: {r:.4f} ({keep})")

    if args.filter_by_corr: 
        if correlations is None: 
            raise ValueError("--filter-by-corr requires --residual-oof")
        keep_names = [n for n, r in correlations.items() if abs(r) > args.residual_threshold]
        if not keep_names: 
            raise ValueError("no features passed the correlation threshold")
        keep_idx = [feature_names.index(n) for n in keep_names]
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]

    mat = {
        "features": X, 
        "labels": y.reshape(-1, 1), 
        "feature_names": np.array(feature_names, dtype="U64"),
        "fips_codes": fips, 
        "disagreement_feature_names": np.array(top_raw_names, dtype="U64")
    }
    savemat(args.out, mat)
    print(f"> Saved {args.out} ({X.shape[0]} rows, {X.shape[1]} features)")
    print("> Final Features:")
    for name in feature_names: 
        print(f"    {name}")
    print("> Disagreement features:", ", ".join(top_raw_names))


if __name__ == "__main__": 
    main() 
