#!/usr/bin/env python3 
# 
# data.py  Andrew Belles  Jan 7th, 2025 
# 
# Dataset Specification and loader creation. 
# 
# 

import numpy as np

from typing import Sequence

from numpy.typing import NDArray

from utils.helpers import (
    bind,
    make_cfg_gap_factory, 
    project_path,
    _mat_str_vector
)

from testbench.utils.paths import (
    LABELS_PATH,
    DATASETS
)

from testbench.utils.transforms import apply_transforms

from testbench.utils.etc        import flatten_imaging

from testbench.utils.paths      import keep_list

from pathlib import Path 

from scipy.io import loadmat, savemat 

from preprocessing.loaders import (
    DatasetDict,
    MetaSpec,
    PassthroughSpec,
    load_spatial_mmap_manifest,
    load_stacking,
    load_tensor_data,
    load_viirs_nchs,
    load_coords_from_mobility,
    load_compact_dataset,
    load_concat_datasets,
    make_oof_dataset_loader,
    load_oof_predictions, 
    ConcatSpec,
    load_spatial_mmap_manifest
)

from preprocessing.loaders import load_passthrough

LEGACY   = ("TIGER",)

PREFIXES = (
    "viirs__",
    "nlcd__",
    "saipe__",
    "tiger__",
    "cross__",
    "coords__",
    "oof__",
    "passthrough__"
)

BASE: dict[str, ConcatSpec] = {
    "VIIRS_MANIFOLD": {
        "name": "VIIRS_MANIFOLD",
        "path": project_path("data", "datasets", "viirs_2020_pooled.mat"),
        "loader": load_compact_dataset
    },
    "SAIPE_2020": {
        "name": "SAIPE", 
        "path": project_path("data", "datasets", "saipe_scalar_2020.mat"),
        "loader": load_compact_dataset
    },
    "SAIPE_2013": {
        "name": "SAIPE", 
        "path": project_path("data", "datasets", "saipe_scalar_2013.mat"),
        "loader": load_compact_dataset
    },
    "SAIPE_MANIFOLD": {
        "name": "SAIPE_MANIFOLD",
        "path": project_path("data", "datasets", "saipe_2020_pooled.mat"),
        "loader": load_compact_dataset
    },
    "USPS_MANIFOLD": {
        "name": "USPS_MANIFOLD",
        "path": project_path("data", "datasets", "usps_2020_pooled.mat"),
        "loader": load_compact_dataset
    },
}

def select_specs_psv(dataset_key: str) -> Sequence[ConcatSpec]: 
    names = dataset_key.split("+")
    return [BASE[n] for n in names]

def override_proba_path(specs, proba_path: str) -> Sequence[ConcatSpec]:
    out = []
    for s in specs: 
        if s["name"].upper() == "OOF": 
            s = dict(s)
            s["path"] = proba_path 
        out.append(s)
    return out 

def dataset_specs(dataset_key: str, proba_path: str): 
    specs = select_specs_psv(dataset_key)
    return override_proba_path(specs, proba_path)

def dataset_loader(specs):
    return load_concat_datasets(
        specs=specs,
        labels_path=LABELS_PATH,
        labels_loader=load_viirs_nchs
    )

def make_dataset_loader(dataset_key: str, proba_path: str): 
    specs = dataset_specs(dataset_key, proba_path)
    return {dataset_key: dataset_loader(specs)}

def load_dataset_raw(dataset_key: str, proba_path: str): 
    specs = dataset_specs(dataset_key, proba_path)
    data  = load_concat_datasets(
        specs=specs,
        labels_path=LABELS_PATH,
        labels_loader=load_viirs_nchs
    )
    X    = data["features"]
    y    = np.asarray(data["labels"]).reshape(-1)
    fips = np.asarray(data["sample_ids"]).astype("U5")
    feature_names = data.get("feature_names")
    return X, y, fips, feature_names

def build_specs(prob_files) -> list[MetaSpec]: 
    specs = []
    for name, prob_path in zip(DATASETS, prob_files): 
        specs.append({
            "name": name.lower(), 
            "proba_path": prob_path,
            "proba_loader": load_oof_predictions
        })
    return specs 

def resolve_expert_loader(dataset_key: str, filter_dir: str | None): 
    base   = BASE[dataset_key]
    klist  = keep_list(filter_dir, dataset_key)
    loader = make_filtered_loader(base["loader"], klist)
    return base, loader 

def resolve_stacking_loader(prob_files, passthrough: bool, filter_dir: str | None): 
    if passthrough: 
        klist = keep_list(filter_dir, "cross")
        return passthrough_loader(prob_files, klist)
    return stacking_loader(prob_files)

def _stacking_loader(_path, *, prob_files): 
    return load_stacking(prob_files)

def stacking_loader(prob_files): 
    return bind(_stacking_loader, prob_files=prob_files)

def _passthrough_loader(_path, *, prob_files, passthrough_specs, passthrough_features):
    data = load_passthrough(
        build_specs(prob_files),
        label_path=LABELS_PATH,
        label_loader=BASE["VIIRS"]["loader"],
        passthrough_specs=passthrough_specs,
        passthrough_features=None 
    )
    transforms = make_cfg_gap_factory(data.get("feature_names"))() 
    if transforms: 
        data["features"] = apply_transforms(data["features"], transforms)
        
        names = np.asarray(data.get("feature_names")) 
        if names is None: 
            names = np.array([], dtype="U64")
        else: 
            names = np.asarray(names, dtype="U64")

        if names.size != data["features"].shape[1]: 
            extra = data["features"].shape[1] - names.size 
            if extra > 0: 
                suffix = (["cfg_gap"] if extra == 1 else 
                [f"cfg_gap_{i+1}" for i in range(extra)])

                data["feature_names"] = np.concatenate(
                    [names, np.asarray(suffix, dtype="U64")]
                )

    return data 

def passthrough_loader(prob_files, passthrough_features = None):
    passthrough_base  = BASE["PASSTHROUGH"]
    passthrough_specs: list[PassthroughSpec] = [
        {
            "name": "cross", 
            "raw_path": passthrough_base["path"],
            "raw_loader": passthrough_base["loader"]
        }
    ] 
    return bind(
        _passthrough_loader,
        prob_files=prob_files,
        passthrough_specs=passthrough_specs,
        passthrough_features=passthrough_features
    )


def binary_loader(data, X, y):
    return {
        "features": X, 
        "labels": y, 
        "coords": data.get("coords"),
        "feature_names": data.get("feature_names"),
        "sample_ids": data.get("sample_ids")
    }

def make_binary_loader(data, label: int): 
    X     = data["features"]
    y     = np.asarray(data["labels"]).reshape(-1)
    y_bin = (y == label).astype(np.int64)
    return binary_loader(data, X, y_bin)

def load_dataset(dataset_key: str): 
    spec = BASE[dataset_key] 
    data = spec["loader"](spec["path"])
    y    = np.asarray(data["labels"]).reshape(-1).astype(np.int64)
    out  = dict(data)
    out["labels"] = y 
    return out 

def load_raw(key: str) -> dict: 
    spec  = BASE[key]
    mat   = loadmat(spec["path"])

    X     = np.asarray(mat["features"], dtype=np.float64)
    fips  = _mat_str_vector(mat["fips_codes"]).astype("U5")
    names = _mat_str_vector(mat["feature_names"]).astype("U64")
    names = np.array([n.strip() for n in names], dtype="U64")

    return {
        "features": X,
        "coords": np.zeros((X.shape[0], 2), dtype=np.float64),
        "feature_names": names, 
        "sample_ids": fips  
    }

def norm_name(name: str) -> str: 
    s = str(name).strip().lower() 
    for prefix in PREFIXES: 
        if s.startswith(prefix): 
            return s[len(prefix):]
    return s

def filter_features(data: dict, keep_list: list[str] | None) -> dict: 
    if not keep_list: 
        return data 
    names = np.asarray(data.get("feature_names"))
    if names is None or names.size == 0: 
        raise ValueError("feature_names missing for filtering")

    keep_norm = {norm_name(n) for n in keep_list}
    cols      = [i for i, n in enumerate(names) if norm_name(n) in keep_norm]
    if not cols: 
        raise ValueError("no features kept after boruta filtering")

    out = dict(data)
    out["features"] = np.asarray(out["features"])[:, cols]
    out["feature_names"] = names[cols]
    return out 

def make_filtered_loader(base_loader, keep_list: list[str] | None): 
    if not keep_list: 
        return base_loader 
    return bind(
        _filtered_loader,
        base_loader=base_loader,
        keep_list=keep_list
    )

def _filtered_loader(path, *, base_loader, keep_list): 
    data = base_loader(path)
    return filter_features(data, keep_list)

def make_tensor_loader(mode, canvas_h, canvas_w, gaf_size): 
    return bind(
        _tensor_loader,
        mode=mode,
        canvas_h=canvas_h,
        canvas_w=canvas_w,
        gaf_size=gaf_size
    )

def _tensor_loader(filepath, *, mode, canvas_h, canvas_w, gaf_size): 
    spatial, mask, gaf, labels, fips = load_tensor_data(filepath)
    X = flatten_imaging(spatial, mask, gaf, mode)
    
    coords = np.zeros((X.shape[0], 2), dtype=np.float64)
    return {
        "features": X, 
        "labels": labels, 
        "coords": coords, 
        "feature_names": np.array([], dtype="U"),
        "sample_ids": fips 
    }

def make_tensor_adapter(mode, canvas_h, canvas_w, gaf_size): 
    n_pix = canvas_h * canvas_w 
    g_pix = gaf_size * gaf_size 

    return bind(
        _tensor_adapter,
        mode=mode,
        canvas_h=canvas_h,
        canvas_w=canvas_w,
        gaf_size=gaf_size
    )

def _tensor_adapter(X, *, mode, canvas_h, canvas_w, gaf_size): 
    n_pix = canvas_h * canvas_w 
    g_pix = gaf_size * gaf_size 

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]

    if mode == "spatial": 
        spatial = X[:, :n_pix].reshape(n, 1, canvas_h, canvas_w)
        mask    = X[:, n_pix:2*n_pix].reshape(n, 1, canvas_h, canvas_w)
        return np.concatenate([spatial, mask], axis=1)

    if mode == "gaf": 
        return X[:, :g_pix].reshape(n, 1, gaf_size, gaf_size)

    if mode == "dual": 
        spatial = X[:, :n_pix].reshape(n, 1, canvas_h, canvas_w)
        mask    = X[:, n_pix:2*n_pix].reshape(n, 1, canvas_h, canvas_w)
        x_main  = np.concatenate([spatial, mask], axis=1)
        x_aux   = X[:, 2*n_pix:2*n_pix + g_pix].reshape(n, 1, gaf_size, gaf_size)
        return x_main, x_aux 

    raise ValueError("mode must be spatial/gaf/dual")

def _roi_loader(
    path,
    *,
    canvas_hw,
    cache_mb=None,
    cache_items=None,
    bag_tiles=False,
    tile_hw=(256,256)
):
    return load_spatial_roi_manifest(
        path, 
        canvas_hw=canvas_hw,
        cache_mb=cache_mb,
        cache_items=cache_items,
        bag_tiles=bag_tiles,
        tile_hw=tile_hw
    )


def make_mmap_loader(
    *,
    tile_shape: tuple[int, int, int] = (1, 256, 256), 
    max_bag_size: int = 64, 
    sample_frac: float | None = None, 
    random_state: int = 0, 
    shuffle_tiles: bool = True 
): 
    return bind(
        load_spatial_mmap_manifest,
        tile_shape=tile_shape,
        max_bag_size=max_bag_size,
        sample_frac=sample_frac,
        random_state=random_state,
        shuffle_tiles=shuffle_tiles
    )

def load_embedding_mat(path: str): 
    mat = loadmat(path)
    if "features" not in mat or "labels" not in mat: 
        raise ValueError(f"missing features/labels in {path}")
    X = np.asarray(mat["features"], dtype=np.float64)
    y = np.asarray(mat["labels"]).reshape(-1).astype(np.int64)
    return X, y

def load_spatial_dataset(
    root_dir: str,
    *,
    tile_shape: tuple[int, int, int],
    max_bag_size: int = 64, 
    sample_frac: float | None = None, 
    random_state: int = 0 
): 

    data = make_mmap_loader(
        tile_shape=tile_shape,
        max_bag_size=max_bag_size,
        sample_frac=sample_frac,
        random_state=random_state
    )(root_dir)

    return (
        data["dataset"],
        data["labels"],
        data["sample_ids"],
        data["collate_fn"],
        data["in_channels"]
    )

def _canon_fips(fips: NDArray) -> NDArray[np.str_]: 
    arr = np.asarray(fips).reshape(-1)
    out = []
    for v in arr: 
        s = str(v).strip() 
        if s.isdigit():
            s = s.zfill(5)
        out.append(s)
    return np.asarray(out, dtype="U5")

def save_embedding_artifact(
    out_path: str, 
    *,
    features: NDArray,
    labels: NDArray,
    fips_codes: NDArray,
    model_key: str, 
    fit_train_indices: NDArray,
    feature_names: NDArray | None = None, 
    source_indices: NDArray | None = None, 
    fit_train_fips: NDArray | None = None, 
    source_year: int = 2013, 
    fit_year: int = 2013, 
    source_split: str = "train",
    extra: dict | None = None 
): 

    if source_split not in {"train", "test", "all"}: 
        raise ValueError(f"invalid source_split: {source_split}")

    X    = np.asarray(features, dtype=np.float32)
    y    = np.asarray(labels).reshape(-1).astype(np.int64)
    fips = _canon_fips(fips_codes)
    if X.ndim != 2:
        raise ValueError(f"features must be 2D, got {X.shape}")
    if not (X.shape[0] == y.shape[0] == fips.shape[0]):
        raise ValueError("features/labels/fips row mismatch")

    n, d = X.shape

    src_idx = (
        np.arange(n, dtype=np.int64)
        if source_indices is None
        else np.asarray(source_indices).reshape(-1).astype(np.int64)
    )
    if src_idx.shape[0] != n:
        raise ValueError("source_indices length mismatch")

    tr_idx = np.asarray(fit_train_indices).reshape(-1).astype(np.int64)
    tr_idx = np.unique(tr_idx)
    if tr_idx.size == 0:
        raise ValueError("fit_train_indices is empty")

    if fit_train_fips is None:
        mask = np.isin(src_idx, tr_idx)
        if mask.any():
            tr_fips = np.unique(fips[mask])
        elif np.all((tr_idx >= 0) & (tr_idx < n)):
            tr_fips = np.unique(fips[tr_idx])
        else:
            raise ValueError("cannot derive fit_train_fips from fit_train_indices/source_indices")
    else:
        tr_fips = np.unique(_canon_fips(fit_train_fips))

    if feature_names is None:
        prefix = model_key.split("/")[-1].lower()
        names = np.asarray([f"{prefix}_emb_{i}" for i in range(d)], dtype="U64")
    else:
        names = np.asarray(feature_names).reshape(-1).astype("U64")
        if names.shape[0] != d:
            raise ValueError(f"feature_names length {names.shape[0]} != embed dim {d}")

    meta = {
        "features": X,
        "labels": y.reshape(-1, 1),
        "fips_codes": fips,
        "feature_names": names,
        "source_indices": src_idx,
        "fit_train_indices": tr_idx,
        "fit_train_fips": tr_fips,
        "model_key": np.asarray([model_key], dtype="U128"),
        "source_year": np.asarray([int(source_year)], dtype=np.int64),
        "fit_year": np.asarray([int(fit_year)], dtype=np.int64),
        "source_split": np.asarray([source_split], dtype="U16"),
        "n_counties": np.asarray([n], dtype=np.int64),
        "embed_dim": np.asarray([d], dtype=np.int64),
    }

    if extra:
        meta.update(extra)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    savemat(out_path, meta)
