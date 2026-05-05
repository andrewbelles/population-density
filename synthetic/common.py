#
# common.py  Andrew Belles  May 4th, 2026
#
# Shared utilities for standalone synthetic experiments.
#

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SyntheticBagDataset:
    bags: list[np.ndarray]
    instance_types: list[np.ndarray]
    signal_masks: list[np.ndarray]
    h: np.ndarray
    theta: np.ndarray
    residual: np.ndarray
    baseline: np.ndarray
    target: np.ndarray
    bag_sizes: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class GraphBlock:
    name: str
    weights: np.ndarray
    eigenvalues: np.ndarray
    basis: np.ndarray


@dataclass(frozen=True)
class LearnedGraphResult:
    weights: np.ndarray
    embedding: np.ndarray
    train_loss: float
    source_representation: str


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    if out.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(out, index=False)
        except ImportError:
            frame.to_csv(out.with_suffix(".csv"), index=False)
    elif out.suffix.lower() == ".csv":
        frame.to_csv(out, index=False)
    else:
        raise ValueError(f"unsupported frame output suffix: {out.suffix}")


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def unit_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        raise ValueError("cannot normalize zero vector")
    return arr / norm


def standardize_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sig = np.nanstd(arr, axis=0, keepdims=True)
    return (arr - mu) / np.clip(sig, eps, None)


def _pool_array(
    x: np.ndarray,
    *,
    signal_mask: np.ndarray | None,
    strategy: str,
    top_frac: float,
) -> np.ndarray:
    name = str(strategy).strip().lower()
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError("each bag must be a nonempty 2D array")
    if name == "mean":
        z = x.mean(axis=0)
    elif name == "max":
        z = x.max(axis=0)
    elif name == "meanmax":
        z = np.concatenate([x.mean(axis=0), x.max(axis=0)], axis=0)
    elif name == "top_frac":
        k = int(max(1, np.ceil(float(top_frac) * x.shape[0])))
        idx = np.argpartition(np.abs(x), kth=x.shape[0] - k, axis=0)[-k:, :]
        cols = np.arange(x.shape[1], dtype=np.int64)[None, :]
        z = x[idx, cols].mean(axis=0)
    elif name == "oracle_signal":
        if signal_mask is None:
            raise ValueError("oracle_signal pooling requires a signal mask")
        mask = np.asarray(signal_mask, dtype=bool).reshape(-1)
        if bool(mask.any()):
            z = x[mask].mean(axis=0)
        else:
            z = x.mean(axis=0)
    else:
        raise ValueError(f"unsupported pooling strategy={strategy!r}")
    return np.asarray(z, dtype=np.float64).reshape(-1)


def pool_bags(
    dataset: SyntheticBagDataset,
    *,
    strategy: str,
    top_frac: float = 0.10,
) -> np.ndarray:
    name = str(strategy).strip().lower()
    pooled: list[np.ndarray] = []
    for bag, signal_mask in zip(dataset.bags, dataset.signal_masks, strict=True):
        pooled.append(_pool_array(np.asarray(bag, dtype=np.float64), signal_mask=signal_mask, strategy=name, top_frac=float(top_frac)))
    return np.vstack(pooled).astype(np.float64, copy=False)


def build_pooling_table(
    dataset: SyntheticBagDataset,
    *,
    strategies: Iterable[str],
    top_frac: float,
) -> dict[str, np.ndarray]:
    return {
        str(strategy): standardize_columns(pool_bags(dataset, strategy=str(strategy), top_frac=float(top_frac)))
        for strategy in strategies
    }


def rbf_knn_graph(
    z: np.ndarray,
    *,
    k: int,
    bandwidth_k: int | None = None,
    symmetrize: bool = True,
) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    dist = squareform(pdist(arr, metric="euclidean"))
    w = knn_weight_matrix(dist, k=int(k), bandwidth_k=bandwidth_k)
    if bool(symmetrize):
        w = 0.5 * (w + w.T)
        np.fill_diagonal(w, 0.0)
    return np.asarray(w, dtype=np.float64)


def random_knn_graph(
    n: int,
    *,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coords = rng.normal(size=(int(n), 2))
    return rbf_knn_graph(coords, k=int(k), bandwidth_k=None, symmetrize=True)


def _subsampled_pooling_table(
    dataset: SyntheticBagDataset,
    *,
    strategy: str,
    top_frac: float,
    keep_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    name = str(strategy).strip().lower()
    if name == "oracle_signal":
        raise ValueError("oracle_signal cannot be used for learned graph pooling")
    keep_prob = float(min(max(float(keep_rate), 1e-6), 1.0))
    pooled: list[np.ndarray] = []
    for bag in dataset.bags:
        x = np.asarray(bag, dtype=np.float64)
        keep = rng.random(x.shape[0]) < keep_prob
        if not bool(np.any(keep)):
            keep[int(rng.integers(0, x.shape[0]))] = True
        pooled.append(_pool_array(x[keep], signal_mask=None, strategy=name, top_frac=float(top_frac)))
    return np.vstack(pooled).astype(np.float64, copy=False)


def learned_subsampled_barlow_gsl_graph(
    dataset: SyntheticBagDataset,
    *,
    source_representation: str,
    top_frac: float,
    graph_k: int,
    latent_dim: int,
    hidden_dim: int,
    projector_hidden_dim: int,
    projector_dim: int,
    keep_rate: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    activation: str,
    barlow_lambda: float,
    device: str,
    seed: int,
) -> LearnedGraphResult:
    """Learn a single-modality graph with subsampled Barlow views.

    Each epoch draws two independent instance-subsampled views of every bag,
    pools each view, and trains an encoder/projector with the same invariance
    and decorrelation logic used by Barlow-style applied GSL. The final graph is
    built from the normalized encoder embedding of the full, unsubsampled bags.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "subsampled Barlow learned graph requires torch; run inside the project nix/.venv environment "
            "or disable it with --no-learned-graph-enabled"
        ) from exc

    source_name = str(source_representation).strip().lower()
    base_raw = pool_bags(dataset, strategy=source_name, top_frac=float(top_frac))
    if base_raw.ndim != 2 or base_raw.shape[0] <= 3 or base_raw.shape[1] <= 0:
        raise ValueError("learned GSL input must be a nonempty 2D matrix")
    mu = np.mean(base_raw, axis=0, keepdims=True)
    sig = np.clip(np.std(base_raw, axis=0, keepdims=True), 1e-12, None)

    def norm_view(arr: np.ndarray) -> np.ndarray:
        return ((np.asarray(arr, dtype=np.float64) - mu) / sig).astype(np.float32, copy=False)

    def activation_layer() -> nn.Module:
        mode = str(activation).strip().lower()
        if mode == "relu":
            return nn.ReLU()
        if mode == "tanh":
            return nn.Tanh()
        if mode == "identity":
            return nn.Identity()
        if mode == "gelu":
            return nn.GELU()
        raise ValueError(f"unsupported learned graph activation={activation!r}")

    class BarlowEncoder(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.LayerNorm(int(in_dim)),
                nn.Linear(int(in_dim), int(hidden_dim)),
                activation_layer(),
                nn.Linear(int(hidden_dim), int(latent_dim)),
            )
            self.projector = nn.Sequential(
                nn.LayerNorm(int(latent_dim)),
                nn.Linear(int(latent_dim), int(projector_hidden_dim)),
                activation_layer(),
                nn.Linear(int(projector_hidden_dim), int(projector_dim)),
            )

        def forward(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            raw = self.encoder(x_t)
            return F.normalize(raw, dim=1), self.projector(raw)

    def offdiag_flat(x_t: torch.Tensor) -> torch.Tensor:
        n, m = x_t.shape
        if n != m:
            raise ValueError("offdiag expects square matrix")
        return x_t.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_loss(y1: torch.Tensor, y2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        z1 = (y1 - y1.mean(dim=0, keepdim=True)) / torch.clamp(y1.std(dim=0, keepdim=True, unbiased=False), min=float(eps))
        z2 = (y2 - y2.mean(dim=0, keepdim=True)) / torch.clamp(y2.std(dim=0, keepdim=True, unbiased=False), min=float(eps))
        c = torch.matmul(z1.T, z2) / float(max(int(y1.shape[0]), 1))
        on_diag = torch.sum((torch.diagonal(c) - 1.0) ** 2)
        off_diag = torch.sum(offdiag_flat(c) ** 2)
        return on_diag + float(barlow_lambda) * off_diag

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    use_device = str(device).strip().lower()
    if use_device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(use_device)
    model = BarlowEncoder(in_dim=int(base_raw.shape[1])).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_loss = float("nan")
    model.train()
    for _epoch in range(int(max(1, epochs))):
        view_a = norm_view(_subsampled_pooling_table(dataset, strategy=source_name, top_frac=float(top_frac), keep_rate=float(keep_rate), rng=rng))
        view_b = norm_view(_subsampled_pooling_table(dataset, strategy=source_name, top_frac=float(top_frac), keep_rate=float(keep_rate), rng=rng))
        x1 = torch.as_tensor(view_a, dtype=torch.float32, device=torch_device)
        x2 = torch.as_tensor(view_b, dtype=torch.float32, device=torch_device)
        optimizer.zero_grad(set_to_none=True)
        _z1, y1 = model(x1)
        _z2, y2 = model(x2)
        loss = barlow_loss(y1, y2)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())
    model.eval()
    with torch.no_grad():
        full_x = torch.as_tensor(norm_view(base_raw), dtype=torch.float32, device=torch_device)
        z_final, _y_final = model(full_x)
        embedding = np.asarray(z_final.detach().cpu(), dtype=np.float64)
    weights = rbf_knn_graph(embedding, k=int(graph_k), bandwidth_k=int(graph_k), symmetrize=True)
    return LearnedGraphResult(
        weights=np.asarray(weights, dtype=np.float64),
        embedding=np.asarray(embedding, dtype=np.float64),
        train_loss=float(train_loss),
        source_representation=source_name,
    )


def build_graph_block(
    name: str,
    weights: np.ndarray,
    *,
    mem_k: int,
    row_topk: int | None = None,
) -> GraphBlock:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("graph weights must be square")
    k_row = int(row_topk) if row_topk is not None else max(int(mem_k) * 4, 32)
    evals, evecs = build_moran_basis_fast(w, top_k=int(mem_k), row_topk=int(k_row))
    return GraphBlock(name=str(name), weights=w, eigenvalues=evals, basis=evecs)


def knn_weight_matrix(dist: np.ndarray, k: int, bandwidth_k: int | None = None, eps: float = 1e-9) -> np.ndarray:
    n = int(dist.shape[0])
    if dist.shape[1] != n:
        raise ValueError("dist must be square")
    work = np.asarray(dist, dtype=np.float64).copy()
    np.fill_diagonal(work, np.inf)
    k_eff = int(max(1, min(int(k), n - 1)))
    idx = np.argpartition(work, kth=k_eff - 1, axis=1)[:, :k_eff]
    dsel = np.take_along_axis(work, idx, axis=1)
    if bandwidth_k is None:
        finite = dsel[np.isfinite(dsel)]
        bw = float(np.median(finite)) if finite.size else 1.0
    else:
        kb = min(max(int(bandwidth_k), 1), k_eff)
        kth = np.partition(dsel, kth=kb - 1, axis=1)[:, kb - 1]
        finite = kth[np.isfinite(kth)]
        bw = float(np.median(finite)) if finite.size else 1.0
    if (not np.isfinite(bw)) or bw <= eps:
        bw = 1.0
    w = np.zeros((n, n), dtype=np.float64)
    wsel = np.exp(-np.square(dsel / bw))
    wsel[~np.isfinite(wsel)] = 0.0
    np.put_along_axis(w, idx, wsel, axis=1)
    rs = np.clip(w.sum(axis=1, keepdims=True), eps, None)
    return w / rs


def _row_topk_sparsify_symmetric(w: np.ndarray, k_row: int) -> csr_matrix:
    n = int(w.shape[0])
    if k_row >= n - 1:
        ws = 0.5 * (np.asarray(w, dtype=np.float64) + np.asarray(w, dtype=np.float64).T)
        return csr_matrix(ws)
    k_eff = int(max(1, min(int(k_row), n - 1)))
    work = np.asarray(w, dtype=np.float64).copy()
    np.fill_diagonal(work, -np.inf)
    idx = np.argpartition(work, kth=n - k_eff, axis=1)[:, -k_eff:]
    rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    cols = idx.reshape(-1)
    vals = work[rows, cols]
    vals[~np.isfinite(vals)] = 0.0
    s = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    s = 0.5 * (s + s.T)
    s.setdiag(0.0)
    s.eliminate_zeros()
    return s


def build_moran_basis_fast(w: np.ndarray, top_k: int, eps: float = 1e-9, row_topk: int = 96) -> tuple[np.ndarray, np.ndarray]:
    n = int(w.shape[0])
    if w.shape[1] != n:
        raise ValueError("w must be square")
    if n <= 3:
        raise ValueError("Moran basis requires at least four nodes")
    k_target = int(max(1, min(int(top_k), n - 2)))
    s = _row_topk_sparsify_symmetric(w, k_row=int(row_topk))
    one_over_n = 1.0 / float(n)

    def center(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float64)
        return vv - np.sum(vv) * one_over_n

    def matvec(v: np.ndarray) -> np.ndarray:
        cv = center(v)
        y = s @ cv
        return center(np.asarray(y, dtype=np.float64))

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    k_req = int(min(max(k_target + 8, 2 * k_target), n - 2))
    evals, evecs = eigsh(op, k=k_req, which="LA", tol=1e-3, maxiter=max(300, 4 * n))
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    mask = evals > float(eps)
    if not np.any(mask):
        raise RuntimeError("no positive eigenvalues for Moran basis")
    pevals = evals[mask]
    pevecs = evecs[:, mask]
    k = int(min(k_target, pevecs.shape[1]))
    return pevals[:k], pevecs[:, :k]


def upper_triangle_values(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    idx = np.triu_indices(arr.shape[0], k=1)
    return np.asarray(arr[idx], dtype=np.float64)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return float("nan")
    if float(np.nanstd(aa[mask])) <= 1e-12 or float(np.nanstd(bb[mask])) <= 1e-12:
        return float("nan")
    return float(spearmanr(aa[mask], bb[mask]).statistic)


def residual_moran_score(weights: np.ndarray, residual: np.ndarray, eps: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    rc = r - float(np.nanmean(r))
    denom = float(rc @ rc)
    if denom <= eps:
        return float("nan")
    ws = 0.5 * (w + w.T)
    np.fill_diagonal(ws, 0.0)
    scale = float(np.sum(np.abs(ws)))
    if scale <= eps:
        return float("nan")
    return float((rc @ ws @ rc) / denom)


def full_sample_projection_r2(features: np.ndarray, target: np.ndarray, alpha: float = 1.0) -> float:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] <= 0:
        return float("nan")
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray([float(alpha)], dtype=np.float64), fit_intercept=True))
    model.fit(x, y)
    pred = np.asarray(model.predict(x), dtype=np.float64)
    denom = float(np.sum(np.square(y - float(np.mean(y)))))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum(np.square(y - pred)) / denom)


def graph_diagnostics(
    graphs: dict[str, GraphBlock],
    *,
    oracle_weights: np.ndarray,
    residual: np.ndarray,
) -> pd.DataFrame:
    oracle_vec = upper_triangle_values(oracle_weights)
    rows: list[dict[str, object]] = []
    for name, block in graphs.items():
        rows.append(
            {
                "graph": str(name),
                "graph_oracle_weight_spearman": safe_spearman(upper_triangle_values(block.weights), oracle_vec),
                "residual_moran_score": residual_moran_score(block.weights, residual),
                "mem_residual_projection_r2": full_sample_projection_r2(block.basis, residual),
                "mem_k": int(block.basis.shape[1]),
                "positive_eigenvalue_mean": float(np.mean(block.eigenvalues)) if block.eigenvalues.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def repeated_shuffle_splits(
    n: int,
    *,
    n_repeats: int,
    train_frac: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = ShuffleSplit(
        n_splits=int(n_repeats),
        train_size=float(train_frac),
        random_state=int(seed),
    )
    x_dummy = np.zeros((int(n), 1), dtype=np.float64)
    return [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in splitter.split(x_dummy)]


def repeated_kfold_splits(
    n: int,
    *,
    n_trials: int,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    x_dummy = np.zeros((int(n), 1), dtype=np.float64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for trial in range(int(n_trials)):
        splitter = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed) + 1009 * int(trial))
        for tr, te in splitter.split(x_dummy):
            splits.append((np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)))
    return splits


def _fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha_grid: np.ndarray,
    seed: int,
) -> np.ndarray:
    cv_splits = int(min(5, max(2, x_train.shape[0] // 25)))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=int(seed))
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.asarray(alpha_grid, dtype=np.float64), cv=cv, fit_intercept=True),
    )
    model.fit(np.asarray(x_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64).reshape(-1))
    return np.asarray(model.predict(np.asarray(x_test, dtype=np.float64)), dtype=np.float64).reshape(-1)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return float("nan")
    if float(np.std(aa[mask])) <= 1e-12 or float(np.std(bb[mask])) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def evaluate_feature_sets(
    *,
    features: dict[str, np.ndarray | None],
    residual: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha_grid: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        baseline_rmse = float(np.sqrt(np.mean(np.square(r[test_idx]))))
        baseline_mae = float(np.mean(np.abs(r[test_idx])))
        for method, block in features.items():
            if block is None:
                pred_resid = np.zeros(test_idx.shape[0], dtype=np.float64)
                feature_dim = 0
            else:
                x = np.asarray(block, dtype=np.float64)
                if x.ndim != 2 or x.shape[0] != r.shape[0] or x.shape[1] <= 0:
                    raise ValueError(f"invalid feature block for method={method!r}")
                pred_resid = _fit_predict_ridge(
                    x[train_idx],
                    r[train_idx],
                    x[test_idx],
                    alpha_grid=np.asarray(alpha_grid, dtype=np.float64),
                    seed=int(seed) + 7919 * int(fold),
                )
                feature_dim = int(x.shape[1])
            pred_target = b[test_idx] + pred_resid
            resid_err = r[test_idx] - pred_resid
            rmse = float(np.sqrt(np.mean(np.square(resid_err))))
            mae = float(np.mean(np.abs(resid_err)))
            denom = float(np.sum(np.square(r[test_idx])))
            residual_r2 = float(1.0 - np.sum(np.square(resid_err)) / denom) if denom > 1e-12 else float("nan")
            rows.append(
                {
                    "fold": int(fold),
                    "method": str(method),
                    "feature_dim": int(feature_dim),
                    "n_train": int(train_idx.shape[0]),
                    "n_test": int(test_idx.shape[0]),
                    "baseline_residual_rmse": baseline_rmse,
                    "corrected_residual_rmse": rmse,
                    "baseline_residual_mae": baseline_mae,
                    "corrected_residual_mae": mae,
                    "relative_rmse_improvement_pct": 100.0 * (baseline_rmse - rmse) / max(baseline_rmse, 1e-12),
                    "relative_mae_improvement_pct": 100.0 * (baseline_mae - mae) / max(baseline_mae, 1e-12),
                    "residual_r2_vs_zero": residual_r2,
                    "residual_corr_pearson": _safe_corr(r[test_idx], pred_resid),
                    "target_rmse": float(np.sqrt(np.mean(np.square(y[test_idx] - pred_target)))),
                    "target_mae": float(np.mean(np.abs(y[test_idx] - pred_target))),
                }
            )
    fold_df = pd.DataFrame(rows)
    summary = summarize_fold_metrics(fold_df)
    return fold_df, summary


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    eval_col = "eval_id" if "eval_id" in fold_df.columns else "fold"
    summary = (
        fold_df.groupby(["method", "feature_dim"], as_index=False)
        .agg(
            baseline_residual_rmse_mean=("baseline_residual_rmse", "mean"),
            corrected_residual_rmse_mean=("corrected_residual_rmse", "mean"),
            corrected_residual_rmse_std=("corrected_residual_rmse", "std"),
            relative_rmse_improvement_pct_mean=("relative_rmse_improvement_pct", "mean"),
            relative_rmse_improvement_pct_median=("relative_rmse_improvement_pct", "median"),
            relative_rmse_improvement_pct_std=("relative_rmse_improvement_pct", "std"),
            relative_mae_improvement_pct_mean=("relative_mae_improvement_pct", "mean"),
            residual_r2_vs_zero_mean=("residual_r2_vs_zero", "mean"),
            residual_corr_pearson_mean=("residual_corr_pearson", "mean"),
            target_rmse_mean=("target_rmse", "mean"),
            target_mae_mean=("target_mae", "mean"),
            n_folds=(eval_col, "nunique"),
        )
        .sort_values("relative_rmse_improvement_pct_mean", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def attach_graph_names(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    graph_names: list[str | None] = []
    representation_names: list[str | None] = []
    feature_kinds: list[str] = []
    for method in out["method"].astype(str):
        if method == "baseline":
            graph_names.append(None)
            representation_names.append(None)
            feature_kinds.append("baseline")
        elif method == "raw_mem_latent_h_oracle":
            graph_names.append("oracle_latent")
            representation_names.append("latent_h_oracle")
            feature_kinds.append("raw+mem")
        elif method.startswith("raw_mem_"):
            rep = method.removeprefix("raw_mem_")
            graph_names.append(f"{rep}_knn")
            representation_names.append(rep)
            feature_kinds.append("raw+mem")
        elif method.startswith("mem_"):
            graph_names.append(method.removeprefix("mem_"))
            representation_names.append(None)
            feature_kinds.append("mem")
        elif method.startswith("raw_"):
            representation_names.append(method.removeprefix("raw_"))
            graph_names.append(None)
            feature_kinds.append("raw")
        else:
            graph_names.append(None)
            representation_names.append(None)
            feature_kinds.append("other")
    out["graph"] = graph_names
    out["representation"] = representation_names
    out["feature_kind"] = feature_kinds
    return out
