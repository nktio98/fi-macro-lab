"""
Asset manager oversight.

  - factor_regression : OLS of manager excess returns on factor returns
    with Newey-West (HAC) standard errors -- alpha t-stats that survive
    autocorrelated, heteroskedastic return series.
  - rolling_betas     : style-drift monitor (mandate compliance).
  - benjamini_hochberg: false-discovery-rate control across a PANEL of
    managers. With 20+ managers, ~1 will show |t|>2 alpha by pure luck;
    naive t-stat screens systematically over-hire. BH-FDR is the standard
    multiple-testing fix (same logic as the deflated Sharpe in taa.py).
  - appraisal metrics : information ratio, tracking error, hit rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _newey_west_se(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    T, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    u = resid[:, None] * X
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(V))


def factor_regression(excess_ret: pd.Series, factors: pd.DataFrame,
                      nw_lags: int = 6) -> dict:
    """Alpha (annualized, %) and betas with Newey-West t-stats."""
    df = pd.concat([excess_ret, factors], axis=1).dropna()
    y = df.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = _newey_west_se(X, resid, nw_lags)
    tstats = beta / se
    r2 = 1 - resid.var() / y.var()
    out = {"alpha_ann_%": beta[0] * 12 * 100, "alpha_t": tstats[0], "r2": r2}
    for i, f in enumerate(factors.columns, 1):
        out[f"beta_{f}"] = beta[i]
        out[f"t_{f}"] = tstats[i]
    return out


def rolling_betas(excess_ret: pd.Series, factors: pd.DataFrame,
                  window: int = 36) -> pd.DataFrame:
    """Rolling factor betas -- style-drift / mandate-compliance monitor."""
    df = pd.concat([excess_ret, factors], axis=1).dropna()
    out = []
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window:i]
        X = np.column_stack([np.ones(window), w.iloc[:, 1:].to_numpy()])
        b, *_ = np.linalg.lstsq(X, w.iloc[:, 0].to_numpy(), rcond=None)
        out.append(b[1:])
    return pd.DataFrame(out, index=df.index[window - 1:],
                        columns=factors.columns)


def benjamini_hochberg(pvals: pd.Series, fdr: float = 0.10) -> pd.DataFrame:
    """BH step-up procedure. Returns per-manager decision at given FDR."""
    p = pvals.sort_values()
    m = len(p)
    thresh = (np.arange(1, m + 1) / m) * fdr
    below = p.to_numpy() <= thresh
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    reject = pd.Series(False, index=p.index)
    reject.iloc[:k] = True
    return pd.DataFrame({"pval": p.round(4), "bh_threshold": thresh.round(4),
                         "significant_at_FDR": reject}).loc[pvals.index]


def appraisal(excess_ret: pd.Series, benchmark_ret: pd.Series) -> dict:
    active = (excess_ret - benchmark_ret).dropna()
    te = active.std() * np.sqrt(12)
    return {"active_ret_ann_%": round(active.mean() * 12 * 100, 2),
            "tracking_error_%": round(te * 100, 2),
            "info_ratio": round(active.mean() * 12 / te, 2) if te > 0 else 0.0,
            "hit_rate_%": round((active > 0).mean() * 100, 1)}


def alpha_pvalue_from_t(t: float, dof: int) -> float:
    """Two-sided p-value from t-stat (normal approx is fine for dof>60)."""
    from scipy.stats import t as tdist
    return float(2 * (1 - tdist.cdf(abs(t), dof)))


def bootstrap_skill_test(returns: pd.DataFrame, factors: pd.DataFrame,
                         n_boot: int = 1000, seed: int = 0) -> dict:
    """Fama-French (2010) luck-vs-skill bootstrap across a manager panel.

    Each manager's returns are re-simulated under a ZERO-ALPHA null
    (fitted factor exposure + resampled residuals), the whole panel is
    re-estimated per bootstrap, and the cross-section of alpha t-stats
    under pure luck is collected. If the ACTUAL best t-stat sits inside
    the null distribution of best t-stats, your 'star manager' is what
    luck alone produces across this many managers.

    Returns dict(actual_sorted_t, null_percentiles(5/50/95 per rank),
    p_top = P(null max-t >= actual max-t)).
    """
    rng = np.random.default_rng(seed)
    df = pd.concat([returns, factors], axis=1).dropna()
    R = df[returns.columns].to_numpy()
    X = np.column_stack([np.ones(len(df)),
                         df[factors.columns].to_numpy()])
    T, M = R.shape
    beta, *_ = np.linalg.lstsq(X, R, rcond=None)
    resid = R - X @ beta
    fitted_null = X @ np.vstack([np.zeros(M), beta[1:]])   # alpha := 0

    def panel_tstats(Rb):
        b, *_ = np.linalg.lstsq(X, Rb, rcond=None)
        e = Rb - X @ b
        XtX_inv_00 = np.linalg.inv(X.T @ X)[0, 0]
        s2 = (e ** 2).sum(0) / (T - X.shape[1])
        return b[0] / np.sqrt(s2 * XtX_inv_00)

    actual_t = np.sort(panel_tstats(R))
    null_sorted = np.empty((n_boot, M))
    for b in range(n_boot):
        idx = rng.integers(0, T, T)                        # iid time resample
        Rb = fitted_null + resid[idx]
        null_sorted[b] = np.sort(panel_tstats(Rb))
    pct = np.percentile(null_sorted, [5, 50, 95], axis=0)
    p_top = float((null_sorted[:, -1] >= actual_t[-1]).mean())
    return {"actual_sorted_t": actual_t,
            "null_5": pct[0], "null_50": pct[1], "null_95": pct[2],
            "p_top": p_top, "n_boot": n_boot}
