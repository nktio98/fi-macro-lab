"""
Macro nowcasting: monthly activity factor + GDP bridge regression.

Answers "where is the economy NOW?" before quarterly GDP prints:

  1. indicator_panel  : monthly macro indicators per economy (FRED),
                        transformed to stationary yoy/mom growth rates.
  2. activity_factor  : first principal component of the standardized
                        panel, estimated with EM-PCA so missing months
                        (ragged edges, mixed publication lags) are handled
                        instead of dropped -- the core trick of nowcasting.
  3. bridge_nowcast   : regress quarterly GDP growth on the quarter-
                        averaged factor (+ AR(1) term); apply to the
                        current partial quarter -> GDP nowcast.

This is deliberately a small bridge/DFM-lite, not a large BVAR: with
free monthly data for Asia (some series lag or were discontinued) the
honest play is a transparent factor + bridge with Newey-West inference.

Coverage is data-limited: ID / KR / JP / US have usable free series;
SG / MY / TH have none alive on FRED (documented gap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_live import fred_csv
from .managers import _newey_west_se

# economy -> GDP series (quarterly, real, IMF/BEA) + monthly indicators.
# transform: 'yoy' = 12m % change, 'mom' = 1m % change, 'level' = as-is.
SERIES = {
    "ID": {"name": "Indonesia", "gdp": "NGDPRSAXDCIDQ",
           "indicators": {
               "cpi_yoy": ("IDNCPIALLMINMEI", "yoy"),
               "exports_yoy": ("XTEXVA01IDM667S", "yoy"),
               "palm_oil_yoy": ("PPOILUSDM", "yoy"),
               "brent_yoy": ("POILBREUSDM", "yoy"),
           }},
    "KR": {"name": "Korea", "gdp": "NGDPRSAXDCKRQ",
           "indicators": {
               "cpi_yoy": ("KORCPIALLMINMEI", "yoy"),
               "ip_yoy": ("KORPROINDMISMEI", "yoy"),
               "exports_yoy": ("XTEXVA01KRM667S", "yoy"),
           }},
    "JP": {"name": "Japan", "gdp": "NGDPRSAXDCJPQ",
           "indicators": {
               "cpi_yoy": ("JPNCPIALLMINMEI", "yoy"),
               "ip_yoy": ("JPNPROINDMISMEI", "yoy"),
               "exports_yoy": ("XTEXVA01JPM667S", "yoy"),
           }},
    "US": {"name": "United States", "gdp": "GDPC1",
           "indicators": {
               "cpi_yoy": ("CPIAUCSL", "yoy"),
               "ip_yoy": ("INDPRO", "yoy"),
               "payrolls_mom": ("PAYEMS", "mom"),
               "retail_yoy": ("RSAFS", "yoy"),
           }},
}


def _transform(s: pd.Series, how: str) -> pd.Series:
    if how == "yoy":
        return s.pct_change(12) * 100
    if how == "mom":
        return s.pct_change(1) * 100
    return s


def indicator_panel(economy: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Monthly transformed indicator panel; series that fail to download
    are skipped (free Asian macro data is patchy -- that's the reality)."""
    spec = SERIES[economy]
    cols = {}
    for label, (sid, how) in spec["indicators"].items():
        try:
            raw = fred_csv([sid], start="1995-01-01")
            s = raw.iloc[:, 0].resample("ME").last()
            cols[label] = _transform(s, how)
        except Exception:
            continue
    if not cols:
        raise ValueError(f"no indicators available for {economy}")
    return pd.DataFrame(cols).loc[start:].dropna(how="all")


def activity_factor(panel: pd.DataFrame, n_iter: int = 50,
                    tol: float = 1e-8):
    """First PC of the standardized panel via EM-PCA (missing-tolerant).

    Returns (factor, loadings): factor is a unit-variance pd.Series whose
    POSITIVE values mean above-trend activity; loadings per indicator."""
    Z = (panel - panel.mean()) / panel.std()
    M = Z.to_numpy()
    miss = np.isnan(M)
    X = np.where(miss, 0.0, M)
    prev = np.inf
    for _ in range(n_iter):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        approx = S[0] * np.outer(U[:, 0], Vt[0])
        X = np.where(miss, approx, M)
        delta = np.abs(S[0] - prev)
        if delta < tol:
            break
        prev = S[0]
    f = U[:, 0] * S[0]
    load = Vt[0]
    # sign convention: factor up = indicators up (activity strong)
    if np.nansum(load) < 0:
        f, load = -f, -load
    factor = pd.Series(f / f.std(), index=panel.index, name="activity")
    loadings = pd.Series(load, index=panel.columns, name="loading")
    return factor, loadings


def gdp_growth(economy: str, start: str = "2000-01-01") -> pd.Series:
    """Quarterly real GDP growth, q/q %, from FRED."""
    gdp = fred_csv([SERIES[economy]["gdp"]], start=start).iloc[:, 0].dropna()
    g = np.log(gdp).diff() * 100
    g.index = g.index.to_period("Q").to_timestamp("Q")   # quarter-end stamps
    return g.rename("gdp_qoq_pct").dropna()


def bridge_nowcast(economy: str, start: str = "2000-01-01") -> dict:
    """Bridge regression: GDP growth on quarter-averaged activity factor.

    gdp_g(q) = a + b*factor_avg(q) + c*gdp_g(q-1) + e,  Newey-West SEs.
    The nowcast applies the fit to the latest (possibly partial) quarter's
    factor average -- months publish ahead of the GDP print, which is the
    entire point of nowcasting."""
    panel = indicator_panel(economy, start)
    factor, loadings = activity_factor(panel)
    g = gdp_growth(economy, start)

    fq = factor.resample("QE").mean().rename("factor_q")
    df = pd.concat([g, fq], axis=1)
    df["gdp_lag"] = df["gdp_qoq_pct"].shift(1)
    est = df.dropna()
    X = np.column_stack([np.ones(len(est)), est["factor_q"], est["gdp_lag"]])
    y = est["gdp_qoq_pct"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = _newey_west_se(X, resid, lags=2)
    r2 = 1 - resid.var() / y.var()

    # nowcast: quarters AFTER the last GDP print where the factor exists
    pending = df[(df.index > g.index.max()) & df["factor_q"].notna()]
    nowcasts = {}
    last_gdp = g.iloc[-1]
    for q, row in pending.iterrows():
        val = beta[0] + beta[1] * row["factor_q"] + beta[2] * last_gdp
        nowcasts[q] = float(val)
        last_gdp = val                      # chain forward if 2 pending qs
    return {"economy": SERIES[economy]["name"], "factor": factor,
            "loadings": loadings, "panel": panel, "gdp_growth": g,
            "beta": beta, "t_stats": beta / se, "r2": float(r2),
            "nowcast": nowcasts,
            "latest_factor": float(factor.iloc[-1]),
            "factor_date": factor.index[-1]}
