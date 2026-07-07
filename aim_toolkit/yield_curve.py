"""
Dynamic Nelson-Siegel (Diebold-Li) yield curve engine.

Pipeline:
  1. Cross-sectional fit: for each date, OLS of yields on NS loadings
     (lambda chosen by grid search over the full panel).
  2. Time-series dynamics: VAR(1) on the level/slope/curvature factors.
  3. Forecast: iterate the VAR forward h steps, reconstruct the curve.

Upgrade paths (interfaces kept compatible):
  - AFNS: add the Christensen-Diebold-Rudebusch yield-adjustment term.
  - ACM term premium: bolt a linear-regression-based affine model on the
    same factor panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ns_loadings(maturities: np.ndarray, lam: float) -> np.ndarray:
    """Nelson-Siegel factor loadings. maturities in years. Returns (n_mat, 3)."""
    tau = np.asarray(maturities, dtype=float)
    x = lam * tau
    slope = (1 - np.exp(-x)) / x
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(tau), slope, curv])


def fit_cross_section(yields: pd.DataFrame, lam: float) -> pd.DataFrame:
    """OLS fit of NS factors for every date. yields: index=date, cols=maturities(yrs)."""
    mats = yields.columns.to_numpy(dtype=float)
    X = ns_loadings(mats, lam)
    # beta = (X'X)^-1 X' y  for all dates at once
    beta, *_ = np.linalg.lstsq(X, yields.to_numpy().T, rcond=None)
    return pd.DataFrame(beta.T, index=yields.index,
                        columns=["level", "slope", "curvature"])


def fit_lambda(yields: pd.DataFrame, grid=None) -> float:
    """Grid-search lambda minimizing total squared fitting error."""
    if grid is None:
        grid = np.linspace(0.2, 2.0, 37)
    mats = yields.columns.to_numpy(dtype=float)
    Y = yields.to_numpy().T  # (n_mat, n_dates)
    best_lam, best_sse = None, np.inf
    for lam in grid:
        X = ns_loadings(mats, lam)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        sse = np.sum((Y - X @ beta) ** 2)
        if sse < best_sse:
            best_lam, best_sse = float(lam), sse
    return best_lam


class VAR1:
    """Minimal VAR(1) with intercept, OLS-estimated equation by equation."""

    def fit(self, F: pd.DataFrame) -> "VAR1":
        Z = F.to_numpy()
        X = np.column_stack([np.ones(len(Z) - 1), Z[:-1]])
        Y = Z[1:]
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self.c = B[0]
        self.A = B[1:].T                      # Y_t = c + A Y_{t-1} + e
        resid = Y - X @ B
        self.Sigma = np.cov(resid.T)
        self.cols = list(F.columns)
        self.last = Z[-1]
        return self

    def forecast(self, h: int) -> pd.DataFrame:
        out, z = [], self.last.copy()
        for _ in range(h):
            z = self.c + self.A @ z
            out.append(z.copy())
        return pd.DataFrame(out, columns=self.cols,
                            index=pd.RangeIndex(1, h + 1, name="h"))

    def long_run_mean(self) -> np.ndarray:
        return np.linalg.solve(np.eye(len(self.c)) - self.A, self.c)


def smith_wilson(tenors: np.ndarray, zero_rates_pct: np.ndarray,
                 ufr: float = 0.038, alpha: float = 0.15):
    """Smith-Wilson curve interpolation/extrapolation (EIOPA / MAS RBC-2
    style): fits the observed zero curve EXACTLY and converges to the
    ultimate forward rate (UFR) beyond the last liquid point.

    tenors in years, zero rates in % (continuous compounding assumed).
    ufr in decimals (as an annual rate; converted to continuous inside).
    alpha = convergence speed. Returns f(t)->zero rate in DECIMALS for
    any t -- drop-in for the liability curve_fn.

    This is the regulator-prescribed answer to 'what discount rate for a
    40y cash flow when the market stops at 30y'."""
    u = np.asarray(tenors, dtype=float)
    z = np.asarray(zero_rates_pct, dtype=float) / 100
    f_inf = np.log(1 + ufr)
    P = np.exp(-z * u)                        # market zero-coupon prices

    def W(t, s):
        t, s = np.asarray(t, float), np.asarray(s, float)
        mn, mx = np.minimum.outer(t, s), np.maximum.outer(t, s)
        return np.exp(-f_inf * np.add.outer(t, s)) * (
            alpha * mn - 0.5 * np.exp(-alpha * mx)
            * (np.exp(alpha * mn) - np.exp(-alpha * mn)))

    zeta = np.linalg.solve(W(u, u), P - np.exp(-f_inf * u))

    def curve_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t = np.where(t <= 0, 1e-6, t)
        price = np.exp(-f_inf * t) + W(t, u) @ zeta
        out = -np.log(np.clip(price, 1e-12, None)) / t
        return out if out.size > 1 else float(out[0])

    return curve_fn


def afns_adjustment(maturities: np.ndarray, lam: float,
                    sig: np.ndarray) -> np.ndarray:
    """AFNS yield-adjustment term (Christensen-Diebold-Rudebusch 2011,
    independent-factor case), in DECIMAL yield units.

    sig = (s1, s2, s3): annualized factor-innovation vols in decimals.
    The arbitrage-free NS yield is the NS yield MINUS this term -- it is
    the convexity correction that pulls long yields down. Grows ~tau^2
    in the level vol, so it matters at 20y-30y and is negligible short."""
    tau = np.asarray(maturities, dtype=float)
    s1, s2, s3 = sig
    L = lam
    e1, e2 = np.exp(-L * tau), np.exp(-2 * L * tau)
    t1 = s1 ** 2 * tau ** 2 / 6
    t2 = s2 ** 2 * (1 / (2 * L ** 2) - (1 - e1) / (L ** 3 * tau)
                    + (1 - e2) / (4 * L ** 3 * tau))
    t3 = s3 ** 2 * (1 / (2 * L ** 2) + e1 / L ** 2 - tau * e2 / (4 * L)
                    - 3 * e2 / (4 * L ** 2)
                    - 2 * (1 - e1) / (L ** 3 * tau)
                    + 5 * (1 - e2) / (8 * L ** 3 * tau))
    return t1 + t2 + t3


class DNSModel:
    """End-to-end Dynamic Nelson-Siegel model."""

    def fit(self, yields: pd.DataFrame) -> "DNSModel":
        self.maturities = yields.columns.to_numpy(dtype=float)
        self.lam = fit_lambda(yields)
        self.factors = fit_cross_section(yields, self.lam)
        self.var = VAR1().fit(self.factors)
        fitted = self.reconstruct(self.factors)
        self.rmse_bp = float(np.sqrt(np.mean(
            (fitted.to_numpy() - yields.to_numpy()) ** 2)) * 100)
        return self

    def reconstruct(self, factors: pd.DataFrame) -> pd.DataFrame:
        X = ns_loadings(self.maturities, self.lam)
        return pd.DataFrame(factors.to_numpy() @ X.T,
                            index=factors.index, columns=self.maturities)

    def forecast_curve(self, h: int) -> pd.DataFrame:
        return self.reconstruct(self.var.forecast(h))


class AFNSModel(DNSModel):
    """DNS + arbitrage-free yield adjustment (two-step approximation).

    Fits DNS, estimates the factor-innovation vols from the VAR(1)
    residuals (annualized, decimals), and subtracts the CDR closed-form
    convexity term when reconstructing. Not full Kalman-MLE AFNS, but
    captures the first-order arbitrage-free correction on the long end.
    Assumes a monthly panel (annualization factor sqrt(12))."""

    def fit(self, yields: pd.DataFrame) -> "AFNSModel":
        super().fit(yields)
        resid_sd = np.sqrt(np.diag(self.var.Sigma))
        self.sig = resid_sd / 100 * np.sqrt(12)     # % monthly -> dec annual
        self.adj_pct = afns_adjustment(self.maturities, self.lam,
                                       self.sig) * 100
        fitted = self.reconstruct(self.factors)
        self.rmse_bp = float(np.sqrt(np.mean(
            (fitted.to_numpy() - yields.to_numpy()) ** 2)) * 100)
        return self

    def reconstruct(self, factors: pd.DataFrame) -> pd.DataFrame:
        base = super().reconstruct(factors)
        return base - self.adj_pct if hasattr(self, "adj_pct") else base


class ACMTermPremium:
    """Adrian-Crump-Moench (2013) linear-regression term premium.

    Three-step estimator on a monthly yield panel (columns = maturities
    in YEARS, yields in %):
      1. K principal components of yields -> pricing factors X_t; VAR(1)
         gives innovations v and Sigma.
      2. Regress bond excess returns on [1, X_{t-1}, v_t] -> risk
         exposures beta; cross-sectional step -> prices of risk
         (lambda0, lambda1).
      3. Affine recursions give model yields; rerunning them with
         lambda = 0 gives RISK-NEUTRAL yields (pure expectations).
         Term premium = model yield - risk-neutral yield.

    Yields are interpolated to a monthly-maturity grid internally; the
    1-period rate is proxied by the shortest column / 12.
    """

    def fit(self, yields: pd.DataFrame, k_factors: int = 3,
            max_years: int = 10) -> "ACMTermPremium":
        Y = yields.dropna()
        mats_y = Y.columns.to_numpy(dtype=float)
        T = len(Y)
        # -- 1. factors + VAR
        Yc = Y.to_numpy() - Y.to_numpy().mean(0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
        X = U[:, :k_factors] * S[:k_factors]            # (T, K)
        Xlag, Xcur = X[:-1], X[1:]
        Zv = np.column_stack([np.ones(T - 1), Xlag])
        B_var, *_ = np.linalg.lstsq(Zv, Xcur, rcond=None)
        mu, Phi = B_var[0], B_var[1:].T
        V = Xcur - Zv @ B_var                           # innovations (T-1,K)
        Sigma = V.T @ V / (T - 1)
        # -- short rate (decimal, per month) + delta regression
        r = Y.iloc[:, 0].to_numpy() / 100 / 12
        Zd = np.column_stack([np.ones(T), X])
        d, *_ = np.linalg.lstsq(Zd, r, rcond=None)
        delta0, delta1 = d[0], d[1:]
        # -- 2. excess returns on the month grid
        grid = np.arange(12, max_years * 12 + 1, 12)    # months
        def logp(row, n):                                # log price, n months
            y_n = np.interp(n / 12, mats_y, row)
            return -(n / 12) * y_n / 100
        RX = np.empty((T - 1, len(grid)))
        Yv = Y.to_numpy()
        for j, n in enumerate(grid):
            p_now = np.array([logp(Yv[t], n) for t in range(T - 1)])
            p_next = np.array([logp(Yv[t + 1], n - 1) for t in range(T - 1)])
            RX[:, j] = p_next - p_now - r[:-1]
        # regress rx on [1, X_{t-1}, v_t]
        Zr = np.column_stack([np.ones(T - 1), Xlag, V])
        coef, *_ = np.linalg.lstsq(Zr, RX, rcond=None)
        a = coef[0]                                     # (N,)
        C = coef[1:1 + k_factors].T                     # (N, K)
        Bmat = coef[1 + k_factors:].T                   # (N, K)
        E = RX - Zr @ coef
        sigma2 = float(np.mean(E ** 2))
        Bstar = np.array([np.outer(b, b).ravel() for b in Bmat])
        lam0 = np.linalg.pinv(Bmat) @ (a + 0.5 * (Bstar @ Sigma.ravel()
                                                  + sigma2))
        lam1 = np.linalg.pinv(Bmat) @ C
        # -- 3. affine recursions
        def recurse(l0, l1):
            A_, B_ = 0.0, np.zeros(k_factors)
            A_out = np.zeros(max_years * 12 + 1)
            B_out = np.zeros((max_years * 12 + 1, k_factors))
            for n in range(1, max_years * 12 + 1):
                A_ = A_ + B_ @ (mu - l0) \
                    + 0.5 * (B_ @ Sigma @ B_ + sigma2) - delta0
                B_ = B_ @ (Phi - l1) - delta1
                A_out[n], B_out[n] = A_, B_
            return A_out, B_out
        A_p, B_p = recurse(lam0, lam1)
        A_q, B_q = recurse(np.zeros_like(lam0), np.zeros_like(lam1))
        self._X, self._grid, self._idx = X, grid, Y.index
        self._Ap, self._Bp, self._Aq, self._Bq = A_p, B_p, A_q, B_q
        self.mats_years = mats_y
        self._yields = Y
        return self

    def _yield_from(self, A, B, n_months: int) -> pd.Series:
        y = -(A[n_months] + self._X @ B[n_months]) / (n_months / 12) * 100
        return pd.Series(y, index=self._idx)

    def fitted_yield(self, n_years: int = 10) -> pd.Series:
        return self._yield_from(self._Ap, self._Bp, n_years * 12)

    def risk_neutral_yield(self, n_years: int = 10) -> pd.Series:
        """Average expected short rate over n years (expectations part)."""
        return self._yield_from(self._Aq, self._Bq, n_years * 12)

    def term_premium(self, n_years: int = 10) -> pd.Series:
        return (self.fitted_yield(n_years)
                - self.risk_neutral_yield(n_years)).rename(f"tp_{n_years}y")

    def fit_rmse_bp(self, n_years: int = 10) -> float:
        actual = pd.Series(
            [np.interp(n_years, self.mats_years, row)
             for row in self._yields.to_numpy()], index=self._idx)
        return float(np.sqrt(np.mean(
            (self.fitted_yield(n_years) - actual) ** 2)) * 100)
