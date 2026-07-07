"""
FX analytics for an insurance investor (Asian entity, USD assets).

Components:

1. Hedge-cost engine (covered interest parity + cross-currency basis).
   Annualized cost of hedging USD exposure back to local currency:
       hedge_cost ~= r_usd - r_local - xccy_basis
   (basis quoted on the non-USD leg; negative basis makes hedging USD
   assets MORE expensive for the Asian investor).
   Hedged yield pickup = usd_asset_yield - hedge_cost - local_asset_yield.
   This single number drives the "buy hedged USD credit vs local bonds"
   decision that dominates Asian insurance portfolio construction.

2. Fair-value engine: Engle-Granger cointegration + error-correction model.
   Long-run: log spot regressed on fundamentals (rate differential, terms
   of trade proxy, ...). ADF test on residuals (from scratch), ECM speed
   of adjustment -> half-life of misvaluation. This is the workhorse
   behind BEER-style FX valuation used on macro desks.

3. Minimum-variance hedge ratio: rolling OLS of unhedged asset returns
   (local ccy) on FX returns; h* = -beta. Modern practice treats the
   hedge ratio as a time-varying estimate, not a fixed policy number
   (upgrade path: DCC-GARCH conditional betas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------- 1. hedge costs
def hedge_cost(r_usd: pd.Series | float, r_local: pd.Series | float,
               basis_bp: pd.Series | float = 0.0):
    """Annualized cost (%) of hedging USD exposure to local currency."""
    return r_usd - r_local - basis_bp / 100.0


def hedged_pickup(usd_asset_yield, r_usd, r_local, local_asset_yield,
                  basis_bp=0.0):
    """Hedged yield pickup (%) of USD asset vs local asset."""
    return usd_asset_yield - hedge_cost(r_usd, r_local, basis_bp) \
        - local_asset_yield


def pickup_table(markets: dict) -> pd.DataFrame:
    """markets: name -> dict(usd_asset_yield, r_usd, r_local,
    local_asset_yield, basis_bp). Returns decision table sorted by pickup."""
    rows = {}
    for name, m in markets.items():
        hc = hedge_cost(m["r_usd"], m["r_local"], m["basis_bp"])
        pu = m["usd_asset_yield"] - hc - m["local_asset_yield"]
        rows[name] = {
            "USD asset yld": m["usd_asset_yield"],
            "hedge cost": round(hc, 2),
            "hedged USD yld": round(m["usd_asset_yield"] - hc, 2),
            "local asset yld": m["local_asset_yield"],
            "pickup (bp)": round(pu * 100, 0),
        }
    return pd.DataFrame(rows).T.sort_values("pickup (bp)", ascending=False)


# ----------------------------------------------- 2. cointegration / ECM
def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    T, k = X.shape
    sigma2 = resid @ resid / (T - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return beta, se, resid


def adf_tstat(u: np.ndarray, lags: int = 1) -> float:
    """ADF t-stat (no constant; u is a residual, mean ~ 0) on du = rho*u(-1)+..."""
    du = np.diff(u)
    X = [u[lags:-1] if lags else u[:-1]]
    du_dep = du[lags:]
    for j in range(1, lags + 1):
        X.append(du[lags - j:-j])
    X = np.column_stack(X)
    beta, se, _ = _ols(X, du_dep)
    return float(beta[0] / se[0])


# Engle-Granger critical values (2 variables, constant in coint. regression)
EG_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


class ECMFairValue:
    """Engle-Granger two-step: long-run fair value + error-correction dynamics."""

    def fit(self, log_spot: pd.Series, fundamentals: pd.DataFrame) -> "ECMFairValue":
        y = log_spot.to_numpy()
        F = fundamentals.to_numpy()
        X = np.column_stack([np.ones(len(y)), F])
        self.beta, _, u = _ols(X, y)
        self.resid = pd.Series(u, index=log_spot.index, name="misvaluation")
        self.adf_t = adf_tstat(u, lags=1)
        self.cointegrated_5pct = self.adf_t < EG_CRIT[0.05]
        # ECM: d(log_spot) = a + gamma * u(-1) + phi * d(log_spot)(-1)
        dy = np.diff(y)
        Xe = np.column_stack([np.ones(len(dy) - 1), u[1:-1], dy[:-1]])
        be, se, _ = _ols(Xe, dy[1:])
        self.gamma, self.gamma_t = float(be[1]), float(be[1] / se[1])
        self.half_life = float(np.log(0.5) / np.log(1 + self.gamma)) \
            if -1 < self.gamma < 0 else np.inf
        self.cols = list(fundamentals.columns)
        return self

    def fair_value(self, fundamentals: pd.DataFrame) -> pd.Series:
        X = np.column_stack([np.ones(len(fundamentals)),
                             fundamentals.to_numpy()])
        return pd.Series(X @ self.beta, index=fundamentals.index)

    def summary(self) -> str:
        sig = "YES" if self.cointegrated_5pct else "NO"
        return (f"ADF t-stat on residual: {self.adf_t:.2f} "
                f"(5% crit {EG_CRIT[0.05]}) -> cointegrated: {sig}\n"
                f"ECM speed gamma: {self.gamma:.3f} (t={self.gamma_t:.1f}) "
                f"-> half-life of misvaluation: {self.half_life:.1f} periods")


# --------------------------------------- 3. minimum-variance hedge ratio
def min_var_hedge_ratio(asset_ret_local: pd.Series, fx_ret: pd.Series,
                        window: int = 126) -> pd.Series:
    """Rolling h* = Cov(unhedged_ret, fx_ret)/Var(fx_ret). h*=1 -> full hedge.
    asset_ret_local: UNHEDGED asset return measured in the investor's ccy."""
    cov = asset_ret_local.rolling(window).cov(fx_ret)
    var = fx_ret.rolling(window).var()
    return (cov / var).rename("min_var_hedge_ratio")


# ------------------------------------------ 4. GARCH(1,1) + DCC hedging
def garch11(returns: np.ndarray, n_restarts: int = 2) -> dict:
    """GARCH(1,1) by Gaussian MLE (from scratch, scipy L-BFGS-B).

    r_t = sqrt(h_t) z_t;  h_t = omega + alpha r_{t-1}^2 + beta h_{t-1}.
    Returns dict(omega, alpha, beta, cond_vol, loglik)."""
    from scipy.optimize import minimize
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    v = r.var()

    def filt(params):
        omega, alpha, beta = params
        h = np.empty(len(r))
        h[0] = v
        for t in range(1, len(r)):
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        return h

    def nll(params):
        h = filt(params)
        if (h <= 0).any():
            return 1e10
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)

    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.80)][:n_restarts]:
        res = minimize(nll, x0=[v * (1 - a0 - b0), a0, b0],
                       method="L-BFGS-B",
                       bounds=[(1e-12, None), (1e-6, 0.5), (1e-6, 0.999)])
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x
    return {"omega": omega, "alpha": alpha, "beta": beta,
            "cond_vol": np.sqrt(filt(best.x)), "loglik": -best.fun}


def dcc_hedge_ratio(asset_ret_local: pd.Series, fx_ret: pd.Series) -> dict:
    """DCC(1,1)-GARCH conditional minimum-variance hedge ratio.

    Two-stage Engle (2002): univariate GARCH(1,1) per series, then DCC
    on the standardized residuals:
        Q_t = (1-a-b) Qbar + a e_{t-1} e_{t-1}' + b Q_{t-1}
    h*_t = rho_t * sigma_asset,t / sigma_fx,t -- the conditional version
    of Cov/Var. Returns dict(hedge_ratio, rho, a, b, garch_params)."""
    from scipy.optimize import minimize
    df = pd.concat([asset_ret_local.rename("a"), fx_ret.rename("f")],
                   axis=1).dropna()
    g1 = garch11(df["a"].to_numpy())
    g2 = garch11(df["f"].to_numpy())
    e = np.column_stack([
        (df["a"] - df["a"].mean()).to_numpy() / g1["cond_vol"],
        (df["f"] - df["f"].mean()).to_numpy() / g2["cond_vol"]])
    T = len(e)
    Qbar = e.T @ e / T

    def rho_path(params):
        a, b = params
        Q = Qbar.copy()
        rho = np.empty(T)
        rho[0] = Qbar[0, 1] / np.sqrt(Qbar[0, 0] * Qbar[1, 1])
        for t in range(1, T):
            Q = (1 - a - b) * Qbar + a * np.outer(e[t - 1], e[t - 1]) + b * Q
            rho[t] = Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1])
        return np.clip(rho, -0.9999, 0.9999)

    def nll(params):
        a, b = params
        if a + b >= 0.999:
            return 1e10
        rho = rho_path(params)
        det = 1 - rho ** 2
        quad = (e[:, 0] ** 2 - 2 * rho * e[:, 0] * e[:, 1]
                + e[:, 1] ** 2) / det
        return 0.5 * np.sum(np.log(det) + quad)

    res = minimize(nll, x0=[0.03, 0.94], method="L-BFGS-B",
                   bounds=[(1e-6, 0.3), (0.5, 0.998)])
    a, b = res.x
    rho = pd.Series(rho_path(res.x), index=df.index, name="dcc_rho")
    hr = pd.Series(rho.to_numpy() * g1["cond_vol"] / g2["cond_vol"],
                   index=df.index, name="dcc_hedge_ratio")
    return {"hedge_ratio": hr, "rho": rho, "a": float(a), "b": float(b),
            "garch_asset": g1, "garch_fx": g2}
