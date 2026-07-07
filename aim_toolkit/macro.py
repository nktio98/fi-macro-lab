"""
Macro shock transmission via local projections (Jorda 2005).

Answers "if X shocks by 1 unit today, what happens to Y over the next
h periods?" -- the impulse-response question -- WITHOUT a structural
VAR. For each horizon h a separate OLS is run:

    y_{t+h} (cumulated) = a_h + b_h * shock_t
                          + sum_j c_j * y_{t-j} + sum_j d_j * shock_{t-j}

b_h traced over h is the impulse response. Newey-West standard errors
(lag length h+1) handle the overlapping-horizon autocorrelation that
local projections mechanically induce.

Why LP instead of a sign-restricted SVAR: identical estimand under
correct specification (Plagborg-Moller & Wolf 2021), far more robust to
misspecification, no identification controversy, and it is just OLS --
auditable line by line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .managers import _newey_west_se


def local_projections(response: pd.Series, shock: pd.Series,
                      horizons: int = 12, lags: int = 3,
                      cumulative: bool = True, ci: float = 0.90
                      ) -> pd.DataFrame:
    """Impulse response of `response` to a 1-unit `shock`.

    response : per-period CHANGE of the variable of interest (e.g. monthly
               log-return in %, monthly bp change). With cumulative=True
               the IRF is the cumulated level response, the usual plot.
    shock    : the impulse series in the units you want the IRF per-unit
               of (e.g. bp change in the US 10y).

    Returns DataFrame indexed by horizon with [beta, se, lo, hi].
    """
    df = pd.concat([response.rename("y"), shock.rename("s")], axis=1).dropna()
    y, s = df["y"], df["s"]
    z = 1.6448536 if abs(ci - 0.90) < 1e-9 else 1.959964
    rows = []
    for h in range(horizons + 1):
        dep = (y.rolling(h + 1).sum().shift(-h) if cumulative
               else y.shift(-h))
        cols = [np.ones(len(df)), s.to_numpy()]
        for j in range(1, lags + 1):
            cols.append(y.shift(j).to_numpy())
            cols.append(s.shift(j).to_numpy())
        X = np.column_stack(cols)
        mask = ~(np.isnan(X).any(axis=1) | dep.isna().to_numpy())
        Xm, ym = X[mask], dep.to_numpy()[mask]
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        resid = ym - Xm @ beta
        se = _newey_west_se(Xm, resid, lags=h + 1)
        rows.append({"h": h, "beta": beta[1], "se": se[1],
                     "lo": beta[1] - z * se[1], "hi": beta[1] + z * se[1]})
    return pd.DataFrame(rows).set_index("h")


def irf_summary(irf: pd.DataFrame) -> dict:
    """Peak response, its horizon, and whether it is significant."""
    peak_h = int(irf["beta"].abs().idxmax())
    peak = irf.loc[peak_h]
    return {"peak_h": peak_h, "peak_beta": float(peak["beta"]),
            "significant": bool(peak["lo"] > 0 or peak["hi"] < 0)}
