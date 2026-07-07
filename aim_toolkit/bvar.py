"""
Bayesian VAR with a Minnesota prior — the scenario generator promised
in the stress-engine upgrade path.

Implementation is the dummy-observation (Banbura-Giannone-Reichlin 2010)
form of the conjugate Normal-inverse-Wishart Minnesota prior: the prior
is encoded as artificial data rows, the posterior is then just OLS on
the augmented sample, and simulation draws (Sigma, B) from the exact
NIW posterior before iterating the VAR forward with Gaussian shocks.

Why a BVAR and not plain OLS-VAR for scenarios: with short samples
(free spread data!) an unrestricted VAR overfits and its simulations
explode; Minnesota shrinkage toward univariate white noise/random walks
keeps the posterior predictive well-behaved — shrinkage is the whole
point, same philosophy as Black-Litterman on the allocation side.

Output plugs into stress.monte_carlo_pnl(method="given") so simulated
factor moves run through the SAME revaluation function as the named
deterministic scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MinnesotaBVAR:
    """BVAR(p) with Minnesota prior via dummy observations.

    lambda1 : overall tightness (smaller = tighter to the prior).
    own_mean: prior mean on the first own lag (0 for stationary changes/
              returns, 1 for persistent levels).
    """

    def __init__(self, lags: int = 1, lambda1: float = 0.2,
                 own_mean: float = 0.0):
        self.p, self.l1, self.own_mean = lags, lambda1, own_mean

    def fit(self, Y: pd.DataFrame) -> "MinnesotaBVAR":
        self.cols = list(Y.columns)
        Z = Y.to_numpy(dtype=float)
        T, k = Z.shape
        p = self.p
        sig = np.array([np.std(np.diff(Z[:, i])) if T > 2 else 1.0
                        for i in range(k)])
        sig = np.clip(sig, 1e-8, None)
        # design matrix: X = [1, y_{t-1}, ..., y_{t-p}]
        X = np.column_stack([np.ones(T - p)]
                            + [Z[p - j - 1:T - j - 1] for j in range(p)])
        Yd = Z[p:]
        # -- Minnesota dummy observations
        rows_x, rows_y = [], []
        for j in range(p):                      # lag-coefficient shrinkage
            for i in range(k):
                x = np.zeros(1 + k * p)
                x[1 + j * k + i] = sig[i] * (j + 1) / self.l1
                y = np.zeros(k)
                if j == 0:
                    y[i] = self.own_mean * sig[i] / self.l1
                rows_x.append(x); rows_y.append(y)
        for i in range(k):                      # residual-covariance prior
            x = np.zeros(1 + k * p)
            y = np.zeros(k); y[i] = sig[i]
            rows_x.append(x); rows_y.append(y)
        Xs = np.vstack([X] + [np.array(rows_x)])
        Ys = np.vstack([Yd] + [np.array(rows_y)])
        # posterior = OLS on augmented data
        XtX = Xs.T @ Xs
        self.B = np.linalg.solve(XtX, Xs.T @ Ys)          # (1+kp, k)
        E = Ys - Xs @ self.B
        self.S = E.T @ E                                   # IW scale
        self.nu = len(Ys) - (1 + k * p)                    # IW dof
        self.XtX_inv = np.linalg.solve(XtX, np.eye(1 + k * p))
        self.k, self.last = k, Z[-p:][::-1].ravel()        # newest first
        return self

    def _draw_params(self, rng):
        """One (Sigma, B) draw from the NIW posterior."""
        k, m = self.k, self.B.shape[0]
        # Sigma ~ IW(S, nu) via inverse of Wishart draw
        L = np.linalg.cholesky(np.linalg.inv(self.S))
        A_ = L @ rng.standard_normal((k, self.nu)) if self.nu > k else None
        W = A_ @ A_.T
        Sigma = np.linalg.inv(W)
        # B | Sigma ~ MN(B, XtX_inv, Sigma)
        chol_row = np.linalg.cholesky(self.XtX_inv + 1e-12 * np.eye(m))
        chol_col = np.linalg.cholesky(Sigma + 1e-12 * np.eye(k))
        Bd = self.B + chol_row @ rng.standard_normal((m, k)) @ chol_col.T
        return Sigma, Bd

    def simulate(self, h: int = 1, n_draws: int = 5000,
                 seed: int = 0) -> pd.DataFrame:
        """Posterior-predictive simulation: for each draw, sample (Sigma,B)
        then iterate the VAR h steps ahead with Gaussian shocks from the
        last observed state. Returns the CUMULATIVE h-step move per draw
        -- ready to feed stress.monte_carlo_pnl(method='given')."""
        rng = np.random.default_rng(seed)
        k, p = self.k, self.p
        out = np.zeros((n_draws, k))
        for d in range(n_draws):
            Sigma, B = self._draw_params(rng)
            cS = np.linalg.cholesky(Sigma + 1e-12 * np.eye(k))
            state = self.last.copy()                       # (k*p,) newest first
            cum = np.zeros(k)
            for _ in range(h):
                x = np.concatenate([[1.0], state])
                y = x @ B + cS @ rng.standard_normal(k)
                cum += y
                state = np.concatenate([y, state[:-k]]) if p > 1 else y
            out[d] = cum
        return pd.DataFrame(out, columns=self.cols)

    def coef_table(self) -> pd.DataFrame:
        idx = ["const"] + [f"{c}.L{j + 1}" for j in range(self.p)
                           for c in self.cols]
        return pd.DataFrame(self.B, index=idx, columns=self.cols).round(3)
