"""
Research extensions beyond the paper:

  decay_profile          -- how FAST does mispricing correct? FMB of
                            cumulative h-month-ahead returns on the
                            residual, h = 1..12, with Newey-West t-stats
                            (overlapping horizons autocorrelate the
                            monthly slopes; plain FMB t's overstate).
  liquidity_double_sort  -- is the premium a limits-to-arbitrage
                            phenomenon? Mispricing quintiles WITHIN
                            dollar-volume terciles, plus turnover and
                            the break-even transaction cost.
  regime_conditional     -- is the premium earned in calm or stressed
                            credit markets? FMB slope series split by a
                            jump-model regime on the market-wide spread
                            (the website's own regime machinery, pointed
                            at the paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aim_toolkit.managers import _newey_west_se
from aim_toolkit.regimes import JumpModel

from .config import IG_LABEL, MIN_OBS_QUINTILES
from .fmb import fama_macbeth
from .portfolios import long_short_stats, quintile_returns


def nw_tstat(series: pd.Series, lags: int) -> float:
    """Newey-West t-stat for the mean of a (possibly autocorrelated)
    series -- regression on a constant with HAC standard errors."""
    x = series.dropna().to_numpy()
    X = np.ones((len(x), 1))
    se = _newey_west_se(X, x - x.mean(), lags=max(lags, 1))
    return float(x.mean() / se[0])


# ----------------------------------------------------------- decay profile
def add_forward_returns(panel: pd.DataFrame,
                        horizons: tuple[int, ...] = (1, 2, 3, 6, 9, 12)
                        ) -> pd.DataFrame:
    """Cumulative return from t+1 to t+h per bond, requiring h
    CONSECUTIVE calendar months (gaps in TRACE coverage disqualify)."""
    p = panel.sort_values(["ISSUE_ID", "DATE"]).copy()
    g = p.groupby("ISSUE_ID")
    logret = np.log1p(p["RET_EOM"])
    cs = logret.groupby(p["ISSUE_ID"]).cumsum()
    mnum = p["DATE"].dt.year * 12 + p["DATE"].dt.month
    for h in horizons:
        cum = np.exp(cs.groupby(p["ISSUE_ID"]).shift(-h) - cs) - 1
        gap_ok = g["DATE"].shift(-h).notna() & (
            (mnum.groupby(p["ISSUE_ID"]).shift(-h) - mnum) == h)
        p[f"fwd_ret_{h}"] = cum.where(gap_ok)
    return p


def decay_profile(panel: pd.DataFrame,
                  horizons: tuple[int, ...] = (1, 2, 3, 6, 9, 12),
                  mis_col: str = "spread_resid_w",
                  segment: str = IG_LABEL) -> pd.DataFrame:
    """FMB slope of cumulative h-month returns on mispricing, per h.
    t_plain is the naive FMB t; t_nw corrects the slope series for the
    h-1 months of mechanical overlap."""
    p = add_forward_returns(panel, horizons)
    seg = p[p["RATING_CLASS"] == segment]
    rows = []
    for h in horizons:
        out = fama_macbeth(seg, y_col=f"fwd_ret_{h}", key_cols=[mis_col])
        if mis_col not in out["summary"].index:
            continue
        slopes = out["betas"][mis_col].dropna()
        rows.append({
            "horizon_m": h, "n_months": len(slopes),
            "coef_cum": float(slopes.mean()),
            "t_plain": float(out["summary"].loc[mis_col, "t_stat"]),
            "t_nw": nw_tstat(slopes, lags=h),
        })
    df = pd.DataFrame(rows).set_index("horizon_m")
    df["marginal"] = df["coef_cum"].diff().fillna(df["coef_cum"])
    return df


# --------------------------------------------------- liquidity double sort
def liquidity_double_sort(panel: pd.DataFrame, liq_col: str = "T_DVolume",
                          mis_col: str = "spread_resid_w",
                          segment: str = IG_LABEL,
                          n_liq: int = 3) -> pd.DataFrame:
    """Assign monthly liquidity terciles, then run the mispricing
    long-short and FMB within each tercile. Limits-to-arbitrage says
    the premium should live in the LOW-liquidity tercile."""
    seg = panel[(panel["RATING_CLASS"] == segment)
                & panel[liq_col].notna() & (panel[liq_col] > 0)
                & panel[mis_col].notna()
                & panel["RET_EOM_next"].notna()].copy()
    seg["liq_bucket"] = (
        seg.groupby("DATE")[liq_col]
        .transform(lambda s: pd.qcut(s, n_liq, labels=False,
                                     duplicates="drop")))
    labels = {0: "low_liq", n_liq - 1: "high_liq"}
    rows = []
    for b in range(n_liq):
        sub = seg[seg["liq_bucket"] == b]
        q = quintile_returns(sub, sort_col=mis_col, rating_class=segment,
                             min_bonds=max(25, MIN_OBS_QUINTILES // 2))
        name = labels.get(b, f"mid_liq_{b}")
        if len(q) < 24:
            continue
        ls = long_short_stats(q)
        fm = fama_macbeth(sub, key_cols=[mis_col])["summary"]
        rows.append({
            "bucket": name, "n_months": ls["n_months"],
            "ls_mean_m": ls["mean_monthly"], "ls_t": ls["t_stat"],
            "ls_sharpe": ls["ann_sharpe"],
            "fmb_coef": float(fm.loc[mis_col, "coef"])
            if mis_col in fm.index else np.nan,
            "fmb_t": float(fm.loc[mis_col, "t_stat"])
            if mis_col in fm.index else np.nan})
    return pd.DataFrame(rows).set_index("bucket")


def strategy_turnover(panel: pd.DataFrame, mis_col: str = "spread_resid_w",
                      segment: str = IG_LABEL) -> dict:
    """Monthly one-way turnover of the Q4 and Q0 legs (fraction of names
    replaced) and the break-even one-way cost in bp."""
    seg = panel[(panel["RATING_CLASS"] == segment)
                & panel[mis_col].notna()
                & panel["RET_EOM_next"].notna()]
    legs = {4: [], 0: []}
    prev = {4: set(), 0: set()}
    turns = []
    ls_rets = []
    for date, df_m in seg.groupby("DATE"):
        if len(df_m) < MIN_OBS_QUINTILES:
            continue
        try:
            q = pd.qcut(df_m[mis_col], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        mean_ret = df_m.groupby(q)["RET_EOM_next"].mean()
        if not {0, 4} <= set(mean_ret.index):
            continue
        ls_rets.append(mean_ret[4] - mean_ret[0])
        month_turn = []
        for k in (4, 0):
            cur = set(df_m.loc[q == k, "ISSUE_ID"])
            if prev[k]:
                month_turn.append(1 - len(cur & prev[k]) / max(len(cur), 1))
            prev[k] = cur
        if month_turn:
            turns.append(np.mean(month_turn))
    turnover = float(np.mean(turns))
    mean_ls = float(np.mean(ls_rets))
    # both legs trade `turnover` of their names one-way each month
    breakeven_bp = mean_ls / (2 * turnover) * 1e4 if turnover > 0 else np.inf
    return {"mean_monthly_ls": mean_ls, "one_way_turnover": turnover,
            "breakeven_cost_bp": float(breakeven_bp),
            "n_months": len(turns)}


# ------------------------------------------------------ regime conditional
def regime_conditional_fmb(panel: pd.DataFrame,
                           mis_col: str = "spread_resid_w",
                           segment: str = IG_LABEL,
                           jump_penalty: float = 3.0) -> dict:
    # NOTE: penalty is tuned for MONTHLY data (218 obs); the daily-data
    # default of ~80 would classify almost nothing as stress.
    """Split the monthly FMB slopes by a 2-state jump-model regime fit
    on the market-average spread (level change + rolling vol). Returns
    per-regime slope means with NW t-stats and the calm-stress spread."""
    mkt = panel.groupby("DATE")["T_Spread"].mean().sort_index()
    # stress = persistent wide-spread spells, not just volatile months:
    # first feature (level) also drives the calm/stress state ordering
    feat = np.column_stack([
        mkt.to_numpy(),
        mkt.diff().rolling(3).std().bfill().fillna(0)])
    jm = JumpModel(jump_penalty=jump_penalty).fit(feat)
    regime = pd.Series(jm.states, index=mkt.index, name="stress")

    out = fama_macbeth(panel[panel["RATING_CLASS"] == segment],
                       key_cols=[mis_col])
    slopes = out["betas"][mis_col].dropna()
    reg = regime.reindex(slopes.index).fillna(0).astype(int)
    rows = {}
    for state, name in ((0, "calm"), (1, "stress")):
        s = slopes[reg == state]
        if len(s) < 10:
            continue
        rows[name] = {"n_months": len(s), "coef": float(s.mean()),
                      "t_nw": nw_tstat(s, lags=3)}
    table = pd.DataFrame(rows).T
    diff_t = np.nan
    if {"calm", "stress"} <= set(table.index):
        d = slopes[reg == 1].mean() - slopes[reg == 0].mean()
        se = np.sqrt(slopes[reg == 1].var(ddof=1) / (reg == 1).sum()
                     + slopes[reg == 0].var(ddof=1) / (reg == 0).sum())
        diff_t = float(d / se)
    return {"table": table, "diff_t_stress_minus_calm": diff_t,
            "regime": regime, "stress_share": float(regime.mean())}
