"""
Market regime detection.

Two estimators, both implemented from scratch:

1. GaussianMS  -- 2-state Markov-switching (Gaussian HMM) estimated with the
   Hamilton filter + EM (Baum-Welch). Classic Hamilton (1989) machinery.

2. JumpModel   -- statistical jump model (Bemporad/Nystrup et al.): k-means-
   style clustering of feature vectors with an explicit penalty on state
   switches, solved exactly per iteration by dynamic programming. In recent
   asset-management literature this tends to produce more persistent, more
   tradable regimes than HMMs, and it is robust to non-Gaussian features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- HMM / MS
class GaussianMS:
    """2-state Gaussian Markov-switching model on a univariate series."""

    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-7):
        self.k, self.n_iter, self.tol = n_states, n_iter, tol

    @staticmethod
    def _pdf(y, mu, sig):
        return np.exp(-0.5 * ((y - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

    def fit(self, y: np.ndarray) -> "GaussianMS":
        y = np.asarray(y, float)
        T, k = len(y), self.k
        # init: split by quantile of |y - median| so state 0 = calm, 1 = stressed
        dev = np.abs(y - np.median(y))
        q = pd.qcut(pd.Series(dev).rank(method="first"), k, labels=False).to_numpy()
        mu = np.array([y[q == i].mean() for i in range(k)])
        sig = np.array([max(y[q == i].std(), 1e-4) for i in range(k)])
        P = np.full((k, k), 0.05 / (k - 1)) + np.eye(k) * (0.95 - 0.05 / (k - 1))
        P = P / P.sum(1, keepdims=True)
        pi = np.full(k, 1 / k)
        ll_old = -np.inf
        for _ in range(self.n_iter):
            B = np.column_stack([self._pdf(y, mu[i], sig[i]) for i in range(k)])
            B = np.clip(B, 1e-300, None)
            # forward
            alpha = np.zeros((T, k)); c = np.zeros(T)
            alpha[0] = pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ P) * B[t]
                c[t] = alpha[t].sum(); alpha[t] /= c[t]
            ll = np.log(c).sum()
            # backward
            beta = np.zeros((T, k)); beta[-1] = 1
            for t in range(T - 2, -1, -1):
                beta[t] = (P @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            xi = np.zeros((k, k))
            for t in range(T - 1):
                num = (alpha[t][:, None] * P) * (B[t + 1] * beta[t + 1])[None, :]
                xi += num / num.sum()
            # M-step
            P = xi / xi.sum(1, keepdims=True)
            pi = gamma[0]
            for i in range(k):
                w = gamma[:, i]
                mu[i] = (w * y).sum() / w.sum()
                sig[i] = max(np.sqrt((w * (y - mu[i]) ** 2).sum() / w.sum()), 1e-5)
            if abs(ll - ll_old) < self.tol:
                break
            ll_old = ll
        # order states by volatility: state 0 = low vol
        order = np.argsort(sig)
        self.mu, self.sigma = mu[order], sig[order]
        self.P = P[np.ix_(order, order)]
        self.smoothed = gamma[:, order]
        self.loglik = ll
        return self

    @property
    def expected_duration(self):
        """Expected regime duration in periods, per state."""
        return 1.0 / (1.0 - np.diag(self.P))


# ------------------------------------------------------------- Jump model
class JumpModel:
    """Statistical jump model: k-means + switching penalty, DP-solved states."""

    def __init__(self, n_states: int = 2, jump_penalty: float = 50.0,
                 n_iter: int = 30, seed: int = 0):
        self.k, self.lam, self.n_iter = n_states, jump_penalty, n_iter
        self.rng = np.random.default_rng(seed)

    def _dp_states(self, cost: np.ndarray) -> np.ndarray:
        """Viterbi-style DP: minimize sum(cost[t,s_t]) + lam * #switches."""
        T, k = cost.shape
        V = cost[0].copy()
        back = np.zeros((T, k), int)
        for t in range(1, T):
            trans = V[:, None] + self.lam * (1 - np.eye(k))
            back[t] = np.argmin(trans, axis=0)
            V = cost[t] + trans[back[t], np.arange(k)]
        s = np.zeros(T, int)
        s[-1] = int(np.argmin(V))
        for t in range(T - 1, 0, -1):
            s[t - 1] = back[t, s[t]]
        return s

    def fit(self, X: np.ndarray) -> "JumpModel":
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        Xs = (X - X.mean(0)) / X.std(0)
        # init centroids from quantiles of first feature
        idx = np.argsort(Xs[:, 0])
        centers = np.array([Xs[idx[int((j + .5) * len(Xs) / self.k)]]
                            for j in range(self.k)])
        for _ in range(self.n_iter):
            cost = ((Xs[:, None, :] - centers[None]) ** 2).sum(-1)
            s = self._dp_states(cost)
            new = np.array([Xs[s == j].mean(0) if (s == j).any() else centers[j]
                            for j in range(self.k)])
            if np.allclose(new, centers):
                break
            centers = new
        # order states by first-feature mean (e.g. volatility): 0 = calm
        order = np.argsort(centers[:, 0])
        remap = np.empty(self.k, int); remap[order] = np.arange(self.k)
        self.states = remap[s]
        self.centers = centers[order]
        return self


def regime_summary(states: np.ndarray, series: pd.Series) -> pd.DataFrame:
    """Per-regime descriptive stats of an outcome series (e.g. returns)."""
    df = pd.DataFrame({"state": states, "x": series.to_numpy()})
    g = df.groupby("state")["x"]
    ann = np.sqrt(252)
    out = pd.DataFrame({
        "freq_%": g.size() / len(df) * 100,
        "mean_ann_%": g.mean() * 252 * 100,
        "vol_ann_%": g.std() * ann * 100,
    })
    return out.round(2)
