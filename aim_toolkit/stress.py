"""
Scenario / stress-testing engine for an insurance-style portfolio.

Portfolio is described by exposures and sensitivities:
  - key-rate durations (KRD) per tenor bucket for rate risk
  - spread duration per credit sleeve
  - equity beta, FX delta

Liabilities are a stylized life book: deterministic cash-flow vector,
discounted on the government curve -> duration gap and economic surplus
sensitivity, the core ALM lens at an insurance investor.

Scenarios are instantaneous shocks: {rates: {tenor: bp}, spreads: bp,
equity: %, fx: %}. A simple square-root-of-sum-of-squares capital proxy
(Solvency-II flavored, illustrative only) aggregates per-risk charges.

Upgrade path: replace deterministic shocks with draws from a BVAR /
GARCH-DCC simulation and feed the same revaluation function (interface
is scenario -> P&L, so the Monte Carlo layer drops in unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Portfolio:
    mv: float                                   # total market value
    krd: dict                                   # tenor(yr) -> key-rate duration (yrs, MV-weighted)
    spread_dur: float                           # credit spread duration (MV-weighted)
    credit_weight: float                        # share of MV in credit
    equity_weight: float
    fx_unhedged_weight: float                   # share of MV in unhedged foreign ccy


@dataclass
class LiabilityBook:
    cashflows: np.ndarray                       # yearly CFs, year 1..N
    def pv_and_duration(self, curve_fn) -> tuple[float, float]:
        t = np.arange(1, len(self.cashflows) + 1, dtype=float)
        y = curve_fn(t)                          # zero rates (decimal) at each t
        d = np.exp(-y * t)
        pv = float((self.cashflows * d).sum())
        dur = float((t * self.cashflows * d).sum() / pv)
        return pv, dur


def asset_pnl(p: Portfolio, scen: dict) -> dict:
    """Instantaneous P&L decomposition for one scenario. Shocks: bp / %."""
    rate_pnl = -sum(p.krd.get(k, 0.0) * scen.get("rates", {}).get(k, 0.0)
                    for k in set(p.krd) | set(scen.get("rates", {}))) / 1e4 * p.mv
    spread_pnl = -p.spread_dur * p.credit_weight * scen.get("spreads_bp", 0.0) / 1e4 * p.mv
    eq_pnl = p.equity_weight * scen.get("equity_pct", 0.0) / 100 * p.mv
    fx_pnl = p.fx_unhedged_weight * scen.get("fx_pct", 0.0) / 100 * p.mv
    total = rate_pnl + spread_pnl + eq_pnl + fx_pnl
    return {"rates": rate_pnl, "spreads": spread_pnl, "equity": eq_pnl,
            "fx": fx_pnl, "total": total, "total_pct": total / p.mv * 100}


def run_scenarios(p: Portfolio, scenarios: dict) -> pd.DataFrame:
    rows = {name: asset_pnl(p, s) for name, s in scenarios.items()}
    return pd.DataFrame(rows).T.round(2)


def duration_gap(p: Portfolio, liab: LiabilityBook, curve_fn,
                 asset_dur: float) -> dict:
    pv_l, dur_l = liab.pv_and_duration(curve_fn)
    surplus = p.mv - pv_l
    # economic surplus sensitivity to a parallel +100bp move
    d_assets = -asset_dur * 0.01 * p.mv
    d_liabs = -dur_l * 0.01 * pv_l
    return {"asset_mv": p.mv, "liab_pv": round(pv_l, 1),
            "asset_dur": asset_dur, "liab_dur": round(dur_l, 2),
            "dur_gap": round(asset_dur - dur_l * pv_l / p.mv, 2),
            "surplus": round(surplus, 1),
            "surplus_chg_+100bp": round(d_assets - d_liabs, 1)}


def liability_krd(liab: LiabilityBook, curve_fn, tenors,
                  bump_bp: float = 1.0) -> dict:
    """Liability key-rate durations via triangular bumps at each tenor.

    Bump weights are 1 at the tenor, decay linearly to 0 at the adjacent
    tenors, and extend flat below the first / beyond the last tenor, so the
    per-tenor bumps sum to a parallel shift and the KRDs sum to the total
    duration. Returns tenor -> KRD (years, on liability PV)."""
    t = np.arange(1, len(liab.cashflows) + 1, dtype=float)
    y = curve_fn(t)
    pv0 = float((liab.cashflows * np.exp(-y * t)).sum())
    ks = sorted(tenors)
    out = {}
    for i, k in enumerate(ks):
        lo = ks[i - 1] if i else None
        hi = ks[i + 1] if i < len(ks) - 1 else None
        left = np.ones_like(t) if lo is None else np.clip((t - lo) / (k - lo), 0, 1)
        right = np.ones_like(t) if hi is None else np.clip((hi - t) / (hi - k), 0, 1)
        w = np.where(t <= k, left, right)
        yb = y + w * bump_bp / 1e4
        pvb = float((liab.cashflows * np.exp(-yb * t)).sum())
        out[k] = (pv0 - pvb) / pv0 / (bump_bp / 1e4)
    return out


def krd_gap(p: Portfolio, liab: LiabilityBook, curve_fn) -> pd.DataFrame:
    """Per-tenor surplus sensitivity: asset KRD vs liability KRD (both in
    years of surplus impact per 100bp at that tenor, on asset MV base)."""
    t = np.arange(1, len(liab.cashflows) + 1, dtype=float)
    y = curve_fn(t)
    pv_l = float((liab.cashflows * np.exp(-y * t)).sum())
    lk = liability_krd(liab, curve_fn, list(p.krd))
    rows = {}
    for k in sorted(p.krd):
        a, l = p.krd[k], lk[k] * pv_l / p.mv       # both on asset-MV base
        rows[k] = {"asset_krd": round(a, 2), "liab_krd_scaled": round(l, 2),
                   "gap": round(a - l, 2),
                   "surplus_chg_+100bp": round(-(a - l) * 0.01 * p.mv, 1)}
    return pd.DataFrame(rows).T.rename_axis("tenor")


# Illustrative Solvency-II-flavored standalone charges (fractions of exposure)
DEFAULT_CHARGES = {"equity": 0.39, "spread_per_dur_yr": 0.012, "fx": 0.25}
DEFAULT_CORR = pd.DataFrame(
    [[1.0, 0.5, 0.25], [0.5, 1.0, 0.25], [0.25, 0.25, 1.0]],
    index=["equity", "spread", "fx"], columns=["equity", "spread", "fx"])


def capital_proxy(p: Portfolio, charges=None, corr: pd.DataFrame | None = None) -> dict:
    """Very simplified market-risk capital aggregation (illustrative only)."""
    ch = charges or DEFAULT_CHARGES
    corr = DEFAULT_CORR if corr is None else corr
    scr = np.array([
        p.equity_weight * p.mv * ch["equity"],
        p.credit_weight * p.mv * p.spread_dur * ch["spread_per_dur_yr"],
        p.fx_unhedged_weight * p.mv * ch["fx"],
    ])
    total = float(np.sqrt(scr @ corr.to_numpy() @ scr))
    return {"scr_equity": round(scr[0], 1), "scr_spread": round(scr[1], 1),
            "scr_fx": round(scr[2], 1), "scr_market_total": round(total, 1),
            "scr_pct_of_mv": round(total / p.mv * 100, 2)}
