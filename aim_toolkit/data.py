"""
Data layer.

load_yield_csv(): plug in real data (FRED, MAS, Bloomberg export). Expected
format: first column date, remaining columns maturities in years, yields in %.

simulate_*(): realistic synthetic data so the whole pipeline runs offline.
The yield simulator generates factor dynamics with a persistent VAR and
NS measurement noise; the market simulator produces regime-switching
returns so the regime detectors have real structure to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .yield_curve import ns_loadings

MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])


def parse_maturity(label) -> float:
    """Maturity label -> years. Accepts 0.25 / '0.25' / '3M' / '3 mo' / '10Y'."""
    s = str(label).strip().upper().replace(" ", "")
    try:
        return float(s)
    except ValueError:
        pass
    for suffix, scale in (("MO", 1 / 12), ("M", 1 / 12), ("YR", 1.0),
                          ("Y", 1.0), ("W", 1 / 52), ("D", 1 / 365)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * scale
    raise ValueError(f"Cannot parse maturity label: {label!r}")


def load_yield_csv(path: str) -> pd.DataFrame:
    """First column date; remaining columns maturity labels; yields in %.
    Maturity headers may be numeric years or strings like '3M'/'10Y'."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [parse_maturity(c) for c in df.columns]
    return df.sort_index()


def simulate_yields(n_months: int = 300, seed: int = 42,
                    maturities: np.ndarray = MATURITIES) -> pd.DataFrame:
    """Monthly yield panel with persistent level/slope/curvature dynamics."""
    rng = np.random.default_rng(seed)
    mean = np.array([3.2, -1.4, -0.8])            # level, slope, curvature (%)
    A = np.array([[0.985, 0.00, 0.00],
                  [0.02, 0.95, 0.00],
                  [0.00, 0.03, 0.90]])
    chol = np.diag([0.22, 0.20, 0.28])
    F = np.zeros((n_months, 3))
    F[0] = mean + np.array([1.0, -0.5, 0.3])
    for t in range(1, n_months):
        F[t] = mean + A @ (F[t - 1] - mean) + chol @ rng.standard_normal(3)
    X = ns_loadings(maturities, lam=0.55)
    Y = F @ X.T + rng.normal(0, 0.03, (n_months, len(maturities)))
    idx = pd.date_range("2001-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(np.maximum(Y, 0.01), index=idx, columns=maturities)


def simulate_market(n_days: int = 2500, seed: int = 7) -> pd.DataFrame:
    """Daily returns + credit spread with two latent regimes (calm/stress)."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.995, 0.005],                 # calm  -> stays calm
                  [0.02, 0.98]])                  # stress persists
    s = np.zeros(n_days, int)
    for t in range(1, n_days):
        s[t] = rng.choice(2, p=P[s[t - 1]])
    mu = np.where(s == 0, 0.0004, -0.0010)        # daily equity drift
    vol = np.where(s == 0, 0.007, 0.022)
    ret = rng.normal(mu, vol)
    spread = np.zeros(n_days)
    spread[0] = 120
    for t in range(1, n_days):
        target = 110 if s[t] == 0 else 320
        spread[t] = spread[t - 1] + 0.03 * (target - spread[t - 1]) \
            + rng.normal(0, 3 if s[t] == 0 else 9)
    idx = pd.bdate_range("2016-01-01", periods=n_days)
    return pd.DataFrame({"equity_ret": ret, "credit_spread_bp": spread,
                         "true_state": s}, index=idx)
