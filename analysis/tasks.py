
import numpy as np 
from numpy.typing import NDArray

from typing import (
        Any, 
        Callable, 
        Iterator, 
        Literal, 
        Mapping, 
        Optional, 
        Tuple, 
        Protocol  
)

from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler 

Task = Literal["regression", "binary", "multiclass"]

@dataclass(frozen=True)
class PreparedData: 
    X_train: NDArray[np.float64] 
    X_test: NDArray[np.float64] 
    y_train: NDArray[np.float64] | NDArray[np.int64] 
    y_test: NDArray[np.float64] | NDArray[np.int64] 
    X_scaler: Any 
    y_state: Any 

class RegressionTask: 
    '''
    We store: 
        Name of task, 
        name of metrics, 
        whether a higher value for that metric is better 
    '''

    name = "regression"
    metric_cols = ("rmse", "r2")
    baseline_metric_cols = ("baseline_rmse", "baseline_r2")
    higher_is_better = {"rmse": False, "r2": True, "baseline_rmse": False, "baseline_r2": True}

    def prepare_fold(
        self,
        *,
        X_train_raw: NDArray[np.float64], 
        X_test_raw: NDArray[np.float64], 
        y_train_raw: NDArray[np.float64], 
        y_test_raw: NDArray[np.float64] 
    ) -> PreparedData: 
        
        X_scaler = StandardScaler().fit(X_train_raw)
        y_scaler = StandardScaler() 

        if y_train_raw.ndim == 1:
            y_scaler.fit(y_train_raw.reshape(-1, 1))
            y_train = np.asarray(
                y_scaler.transform(y_train_raw.reshape(-1, 1)) ,dtype=np.float64).ravel() 
            y_test  = np.asarray(
                y_scaler.transform(y_test_raw.reshape(-1, 1)), dtype=np.float64).ravel()
        else:
            y_scaler.fit(y_train_raw) 
            y_train = np.asarray(y_scaler.transform(y_train_raw), dtype=np.float64) 
            y_test  = np.asarray(y_scaler.transform(y_test_raw), dtype=np.float64) 

        X_train = np.asarray(X_scaler.transform(X_train_raw), dtype=np.float64) 
        X_test  = np.asarray(X_scaler.transform(X_test_raw), dtype=np.float64)

        return PreparedData(
            X_train=X_train, 
            X_test=X_test, 
            y_train=y_train,
            y_test=y_test, 
            X_scaler=X_scaler, 
            y_state=y_scaler 
        )

    def model_kwargs(self, prepared: PreparedData) -> dict[str, Any]:
        return {"y_scaler": prepared.y_state}

    def score_fold(
        self, 
        *, 
        prepared: PreparedData, 
        y_pred: NDArray[np.float64], 
        y_train_raw: NDArray[np.float64], 
        y_test_raw: NDArray[np.float64] 
    ) -> dict[str, float]:  

        y_scaler = prepared.y_state 

        if y_test_raw.ndim == 1: 
            y_hat = np.asarray(
                y_scaler.inverse_transform(y_pred.reshape(-1, 1)), dtype=np.float64).ravel()
            y_true = y_test_raw.reshape(-1)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
            r2   = float(r2_score(y_true, y_hat))

            base = float(np.mean(y_train_raw))
            y_base = np.full_like(y_true, base, dtype=np.float64)
            baseline_rmse = float(np.sqrt(mean_squared_error(y_true, y_base))) 
            baseline_r2   = float(r2_score(y_true, y_base))
        else: 
            y_hat  = np.asarray(y_scaler.inverse_transform(y_pred), dtype=np.float64) 
            y_true = y_test_raw 
            mse_cols = mean_squared_error(y_true, y_hat, multioutput="raw_values")
            rmse = float(np.mean(np.sqrt(np.asarray(mse_cols, dtype=np.float64))))
            r2_cols = r2_score(y_true, y_hat, multioutput="raw_values")
            r2 = float(np.mean(np.asarray(r2_cols, dtype=np.float64)))

            y_train_mean = np.mean(y_train_raw, axis=0)
            y_base = np.tile(y_train_mean.reshape(1, -1), (y_true.shape[0], 1))
            base_mse_cols = mean_squared_error(y_true, y_base, multioutput="raw_values")
            baseline_rmse = float(np.mean(np.sqrt(np.asarray(base_mse_cols, dtype=np.float64))))
            baseline_r2  = float(np.mean(np.asarray(r2_score(y_true,y_base, multioutput="raw_values"), dtype=np.float64)))
        return {
            "rmse": rmse, 
            "r2": r2, 
            "baseline_rmse": baseline_rmse, 
            "baseline_r2": baseline_r2
        } 

    def collect_rows(
        self, 
        *, 
        model_name: str, 
        repeat: int | None, 
        fold: int, 
        test_idx: NDArray[np.int64],
        prepared: PreparedData, 
        y_pred: NDArray[np.float64], 
        y_test_raw: NDArray[np.float64], 
        seed: int, 
        output_names: list[str] | None
    ) -> list[dict[str, Any]]: 

        y_scaler = prepared.y_state
        y_pred = np.asarray(y_pred, dtype=np.float64)

        y_true_raw = np.asarray(y_test_raw, dtype=np.float64)
        rows: list[dict[str, Any]] = []

        # 1D target
        if y_true_raw.ndim == 1:
            # ensure shape (n,1) for inverse_transform
            y_hat = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_true = y_true_raw.reshape(-1)
            residuals = y_true - y_hat

            out_name = output_names[0] if output_names else "y"
            for i, idx in enumerate(test_idx):
                rows.append({
                    "model": model_name,
                    "repeat": -1 if repeat is None else int(repeat),
                    "fold": int(fold),
                    "idx": int(idx),
                    "output": 0,
                    "output_name": str(out_name),
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_hat[i]),
                    "residual": float(residuals[i]),
                    "seed": int(seed),
                })
            return rows

        # 2D (multi-output) target
        if y_true_raw.ndim != 2:
            raise ValueError(f"y_test_raw must be 1D or 2D; got shape {y_true_raw.shape}")

        y_pred_arr = y_pred
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)

        if y_pred_arr.shape != y_true_raw.shape:
            raise ValueError(f"pred shape {y_pred_arr.shape} != y_test_raw shape {y_true_raw.shape}")

        y_hat = y_scaler.inverse_transform(y_pred_arr)
        y_true = y_true_raw
        residuals = y_true - y_hat

        k = int(y_true.shape[1])
        if output_names is not None and len(output_names) != k:
            raise ValueError(f"output_names length ({len(output_names)}) != n_outputs ({k})")

        for j in range(k):
            out_name = output_names[j] if output_names else f"y{j}"
            for i, idx in enumerate(test_idx):
                rows.append({
                    "model": model_name,
                    "repeat": -1 if repeat is None else int(repeat),
                    "fold": int(fold),
                    "idx": int(idx),
                    "output": int(j),
                    "output_name": str(out_name),
                    "y_true": float(y_true[i, j]),
                    "y_pred": float(y_hat[i, j]),
                    "residual": float(residuals[i, j]),
                    "seed": int(seed),
                })

        return rows
