"""
Allocation layer: turning views into portfolios.

  - black_litterman : Bayesian blend of equilibrium (reverse-optimized)
    returns with the strategist's views. Solves the classic problem that
    raw mean-variance amplifies estimation error into extreme weights.

  - EntropyPooling  : Meucci (2008). Start from a prior scenario
    distribution (e.g. historical or simulated joint returns, uniform
    probabilities), impose views as moment constraints (E[x_i] = v), and
    find posterior probabilities minimizing relative entropy KL(q || p).
    Strictly more general than BL: views can be on any moment, any
    function of the scenarios, and the full non-normal distribution is
    preserved. Solved exactly via the convex dual (exponential family).

  - mv_optimize     : long-only mean-variance with risk-aversion and
    optional per-asset caps (SLSQP), consuming either BL or EP outputs.

Combined pipeline: stress module simulates scenarios -> entropy pooling
tilts them to the house view -> optimizer produces the TAA proposal.
That chain IS the modern insurance-investor allocation stack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ------------------------------------------------------ Black-Litterman
def implied_returns(w_mkt: np.ndarray, Sigma: np.ndarray,
                    risk_aversion: float = 2.5) -> np.ndarray:
    """Reverse optimization: equilibrium excess returns Pi = lambda*Sigma*w."""
    return risk_aversion * Sigma @ w_mkt


def black_litterman(Sigma: np.ndarray, w_mkt: np.ndarray, P: np.ndarray,
                    Q: np.ndarray, tau: float = 0.05,
                    omega: np.ndarray | None = None,
                    risk_aversion: float = 2.5):
    """Returns (posterior mean, posterior PREDICTIVE covariance Sigma + A).

    A = cov of the mean estimate; the covariance of returns themselves is
    Sigma + A, which is what a mean-variance optimizer must consume --
    feeding it A alone badly understates risk.
    """
    Pi = implied_returns(w_mkt, Sigma, risk_aversion)
    tS = tau * Sigma
    if omega is None:                       # He-Litterman default
        omega = np.diag(np.diag(P @ tS @ P.T))
    A = np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.inv(omega) @ P)
    mu = A @ (np.linalg.inv(tS) @ Pi + P.T @ np.linalg.inv(omega) @ Q)
    return mu, Sigma + A


# ------------------------------------------------------- Entropy pooling
class EntropyPooling:
    """Minimize KL(q||p) s.t. scenario-moment equality constraints A q = b."""

    def fit(self, scenarios: np.ndarray, views_A: np.ndarray,
            views_b: np.ndarray, prior: np.ndarray | None = None):
        """scenarios: (n_scen, n_assets); views_A: (n_views, n_scen) rows are
        g_k(scenario) values; views_b: target E_q[g_k]. Dual solved by BFGS."""
        n = len(scenarios)
        p = np.full(n, 1 / n) if prior is None else prior
        A, b = np.atleast_2d(views_A), np.atleast_1d(views_b)

        def dual(lam):
            logq = np.log(p) - 1 - lam @ A
            z = np.exp(logq - logq.max())
            # normalized posterior handled via extra normalization constraint:
            return np.log(z.sum()) + logq.max() + lam @ b

        res = minimize(dual, x0=np.zeros(len(b)), method="BFGS")
        lam = res.x
        logq = np.log(p) - 1 - lam @ A
        q = np.exp(logq - logq.max())
        self.q = q / q.sum()
        self.scenarios = scenarios
        self.effective_n = float(1 / np.sum(self.q ** 2))   # ENS diagnostic
        self.kl = float(np.sum(self.q * np.log(self.q / p)))
        return self

    def posterior_moments(self):
        mu = self.q @ self.scenarios
        d = self.scenarios - mu
        Sigma = (self.q[:, None] * d).T @ d
        return mu, Sigma


def view_on_mean(scenarios: np.ndarray, asset_idx: int):
    """Helper: constraint row g(scenario) = scenario[asset_idx]."""
    return scenarios[:, asset_idx]


# ------------------------------------------------------------ optimizer
def mv_optimize(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 2.5,
                w_max: float = 0.5, long_only: bool = True) -> np.ndarray:
    n = len(mu)
    obj = lambda w: -(w @ mu - risk_aversion / 2 * w @ Sigma @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0 if long_only else -w_max, w_max)] * n
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP",
                   bounds=bounds, constraints=cons)
    return res.x
