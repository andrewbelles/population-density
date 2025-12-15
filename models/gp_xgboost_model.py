#!/usr/bin/env python3 
# 
# gp_xgboost_model.py  Andrew Belles  Dec 14th, 2025 
# 
# Exposes a Gradient Boosted, Gaussian Process model to 
# CrossValidator to try and extract additional information 
# from Geospatial information 
# 

import helpers as h 
import numpy as np 

import torch, gpytorch

from xgboost import XGBRFRegressor 
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from typing import Any 

class GPBoost(h.ModelInterface): 

    def __init__(self, *, gpu: bool = False, gp_params: dict[str, Any] | None = None, **xgbrf_params: Any): 
        self.gpu = gpu 
        self.gp_params = gp_params or {}
        self.xgbrf_params = xgbrf_params

    def _gp_residual_predict(self, *, coords_train: NDArray[np.float64], residual_train: NDArray[np.float64],
                             coords_test: NDArray[np.float64], seed: int) -> NDArray[np.float64]: 

        coords_train   = np.asarray(coords_train, dtype=np.float64) 
        coords_test    = np.asarray(coords_test, dtype=np.float64) 
        residual_train = np.asarray(residual_train, dtype=np.float64).reshape(-1) 

        coord_scaler = StandardScaler() 
        coords_train_s = coord_scaler.fit_transform(coords_train)
        coords_test_s  = coord_scaler.transform(coords_test) 

        device = torch.device("cuda" if (self.gpu and torch.cuda.is_available()) else "cpu")
        torch.manual_seed(int(seed)) 
        if device.type == "cuda": 
            torch.cuda.manual_seed_all(int(seed))

        train_x = torch.as_tensor(coords_train_s, dtype=torch.float32, device=device) 
        train_y = torch.as_tensor(residual_train, dtype=torch.float32, device=device)
        test_x  = torch.as_tensor(coords_test_s, dtype=torch.float32, device=device)

        n_train    = int(train_x.shape[0])
        n_inducing = int(self.gp_params.get("n_inducing", 512))
        steps      = int(self.gp_params.get("steps", 500))
        lr         = float(self.gp_params.get("lr", 0.001))
        batch_size = int(self.gp_params.get("batch_size", 1024))

        n_inducing = int(min(max(8, n_inducing), n_train))
        batch_size = int(min(max(32, batch_size), n_train))

        inducing_idx    = torch.randperm(n_train, device=device)[:n_inducing]
        inducing_points = train_x[inducing_idx].clone()  

        class _SVGPModel(gpytorch.models.ApproximateGP): 
            def __init__(self, inducing_points_: torch.Tensor):  
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0) 
                )
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self, 
                    inducing_points_, 
                    variational_distribution, 
                    learn_inducing_locations=True 
                )
                super().__init__(variational_strategy)
                self.mean_module  = gpytorch.means.ConstantMean() 
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
                )
            
            def forward(self, x: torch.Tensor):
                mean_x  = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        model = _SVGPModel(inducing_points).to(device) 
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        model.train()
        likelihood.train() 

        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": likelihood.parameters()}],
            lr=lr 
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train)

        for _ in range(steps): 
            batch_idx = torch.randint(0, n_train, (batch_size,), device=device)
            optimizer.zero_grad() 
            output = model(train_x[batch_idx])
            loss = -mll(output, train_y[batch_idx])
            loss.backward() 
            optimizer.step() 

        model.eval() 
        likelihood.eval() 
        with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
            pred = likelihood(model(test_x)).mean 

        # convert result into ndarray explicitly without copy 
        return pred.detach().cpu().numpy().astype(np.float64, copy=False)  

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]:
        X_train, X_test = features 
        y_train, _      = labels 
        c_train, c_test = coords 
        seed = int(kwargs.get("seed", 0))

        include_coords = bool(self.gp_params.get("include_coords_in_booster", False))
        
        xgbrf_params = {k: h.unwrap_scalar(v) for k, v in self.xgbrf_params.items()}
        xgbrf = h.make_xgb_model(
            XGBRFRegressor,
            gpu=self.gpu, 
            base_params={
                "n_jobs": -1, 
                **xgbrf_params 
            }
        )

        if include_coords: 
            X_tr = np.hstack([np.asarray(c_train, np.float64), np.asarray(X_train, np.float64)])
            X_te = np.hstack([np.asarray(c_test, np.float64), np.asarray(X_test, np.float64)])
        else: 
            X_tr = np.asarray(X_train, dtype=np.float64)
            X_te = np.asarray(X_test, dtype=np.float64) 

        y_tr = np.asarray(y_train, dtype=np.float64).reshape(-1)
        xgbrf.fit(X_tr, y_tr)

        base_tr = np.asarray(xgbrf.predict(X_tr), dtype=np.float64).reshape(-1)
        base_te = np.asarray(xgbrf.predict(X_te), dtype=np.float64).reshape(-1)

        residual_tr = np.asarray(y_train, dtype=np.float64).reshape(-1) - base_tr 

        gp_res_te = self._gp_residual_predict(
            coords_train=np.asarray(c_train, dtype=np.float64),
            residual_train=residual_tr, 
            coords_test=np.asarray(c_test, dtype=np.float64), 
            seed=seed 
        )

        return np.asarray(base_te + gp_res_te, dtype=np.float64)
