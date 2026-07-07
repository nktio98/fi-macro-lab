"""
Model & signal governance: did the signal keep working?

  forecast_eval : accuracy metrics for any directional/point forecast
                  vs realized outcomes (hit rate, RMSE, bias, IC).
  cusum_drift   : two-sided CUSUM on standardized forecast errors --
                  flags structural deterioration (drift) long before a
                  full-sample average would move.
  rolling_ir    : rolling information ratio of a signal's active
                  returns -- the standard signal-decay monitor.

The multiple-testing gate at inception is taa.deflated_sharpe; this
module is the POST-deployment leg of governance the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def forecast_eval(forecast: pd.Series, realized: pd.Series) -> dict:
    """Directional + point accuracy of `forecast` for `realized`."""
    df = pd.concat([forecast.rename("f"), realized.rename("r")],
                   axis=1).dropna()
    err = df["r"] - df["f"]
    both = df[(df["f"] != 0)]
    hit = float((np.sign(both["f"]) == np.sign(both["r"])).mean()) \
        if len(both) else np.nan
    ic = float(df["f"].corr(df["r"]))
    return {"n": len(df), "hit_rate": hit,
            "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()), "ic": ic}


def cusum_drift(errors: pd.Series, k: float = 0.5,
                h: float = 5.0) -> pd.DataFrame:
    """Two-sided CUSUM on standardized errors.

    k = allowance (drifts smaller than k sigmas are ignored),
    h = decision threshold in sigmas. Columns [cusum_up, cusum_dn, flag];
    flag=True marks periods where either side breaches h -- the signal's
    error process has shifted and the model needs re-examination."""
    e = errors.dropna()
    z = (e - e.expanding(min_periods=20).mean().shift(1)) \
        / e.expanding(min_periods=20).std().shift(1)
    z = z.fillna(0.0)
    up = np.zeros(len(z)); dn = np.zeros(len(z))
    for i, zi in enumerate(z.to_numpy()):
        prev_u = up[i - 1] if i else 0.0
        prev_d = dn[i - 1] if i else 0.0
        up[i] = max(0.0, prev_u + zi - k)
        dn[i] = max(0.0, prev_d - zi - k)
    out = pd.DataFrame({"cusum_up": up, "cusum_dn": dn}, index=z.index)
    out["flag"] = (out["cusum_up"] > h) | (out["cusum_dn"] > h)
    return out


def rolling_ir(active_ret: pd.Series, window: int = 252,
               ann: int = ANN) -> pd.Series:
    """Rolling annualized information ratio of active returns."""
    mu = active_ret.rolling(window).mean()
    sd = active_ret.rolling(window).std()
    return (mu / sd * np.sqrt(ann)).rename("rolling_ir")


def decay_report(active_ret: pd.Series, window: int = 252) -> dict:
    """Is the signal decaying? Compare full-sample IR vs recent IR and
    the trend of the rolling IR (per-year slope, OLS on time)."""
    ir = rolling_ir(active_ret, window).dropna()
    if len(ir) < 10:
        return {"full_ir": np.nan, "recent_ir": np.nan,
                "ir_trend_per_year": np.nan, "verdict": "insufficient data"}
    full = float(active_ret.mean() / active_ret.std() * np.sqrt(ANN))
    recent = float(ir.iloc[-1])
    t = np.arange(len(ir)) / ANN
    slope = float(np.polyfit(t, ir.to_numpy(), 1)[0])
    verdict = "DECAYING - review" if (recent < 0 and slope < 0) else \
        "degrading" if slope < -0.25 else "stable"
    return {"full_ir": full, "recent_ir": recent,
            "ir_trend_per_year": slope, "verdict": verdict}
