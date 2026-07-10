"""Tests for the research pipeline — all on SYNTHETIC data (the licensed
WRDS panel never enters the test suite)."""
import numpy as np
import pandas as pd
import pytest

from research.extensions import decay_profile, liquidity_double_sort, \
    regime_conditional_fmb, strategy_turnover
from research.fmb import fama_macbeth, fmb_by_segment, fmb_interaction
from research.panel import link_bonds
from research.portfolios import build_factors, long_short_stats, \
    nw_regression, quintile_returns, two_pass_fmb
from research.signals import add_dd_changes, spread_residuals
from research.synth import synth_panel

rng = np.random.default_rng(9)


@pytest.fixture(scope="module")
def planted():
    panel = synth_panel()
    sig = spread_residuals(panel)
    return panel, sig


# ------------------------------------------------------------- link fix
def _link_fixture():
    bond = pd.DataFrame({
        "ISSUE_ID": [1, 1, 2],
        "CUSIP": ["00000000A"] * 2 + ["00000000B"],
        "DATE": pd.to_datetime(["2005-06-30", "2015-06-30", "2010-01-31"]),
    })
    link = pd.DataFrame({
        "CUSIP": ["00000000A", "00000000A", "00000000B"],
        "PERMNO": [111, 222, 333],
        "link_startdt": pd.to_datetime(["2000-01-01", "2011-01-01",
                                        "2000-01-01"]),
        "link_enddt": pd.to_datetime(["2010-12-31", "2099-12-31",
                                      "2099-12-31"]),
    })
    return bond, link


def test_window_merge_assigns_correct_issuer_over_time():
    bond, link = _link_fixture()
    out = link_bonds(bond, link, mode="window")
    assert len(out) == 3                              # no duplication
    a = out[out["CUSIP"] == "00000000A"].sort_values("DATE")
    assert list(a["PERMNO"]) == [111, 222]            # ownership change
    assert out[out["CUSIP"] == "00000000B"]["PERMNO"].iloc[0] == 333


def test_naive_merge_duplicates_rows_documenting_the_bug():
    bond, link = _link_fixture()
    out = link_bonds(bond, link, mode="naive")
    # CUSIP A maps to two PERMNOs -> each of its bond-months DUPLICATES
    assert len(out) == 5
    assert (out.groupby(["CUSIP", "DATE"]).size().max()) == 2


# ---------------------------------------------------------- first stage
def test_spread_residuals_recover_planted_mispricing(planted):
    panel, sig = planted
    p = sig["panel"]
    # residual should be centered and correlate with the DGP error via
    # the return equation: FMB below is the real check; here sanity only
    assert abs(p["spread_resid"].mean()) < 5e-4
    assert sig["r2"].mean() > 0.3                     # fundamentals load
    fs = sig["first_stage"]
    assert fs.loc["DDCamp", "coef"] == pytest.approx(-0.001, abs=3e-4)
    assert fs.loc["log_size", "coef"] == pytest.approx(-0.002, abs=5e-4)


def test_monthly_winsorize_mode_differs_from_pooled():
    panel = synth_panel(T=24, n_bonds=150, seed=3)
    pooled = spread_residuals(panel, winsorize="pooled")["panel"]
    monthly = spread_residuals(panel, winsorize="monthly")["panel"]
    assert not np.allclose(pooled["spread_resid_w"].to_numpy(),
                           monthly["spread_resid_w"].to_numpy())


# ------------------------------------------------------------------ FMB
def test_fmb_recovers_planted_premium_in_ig_not_hy(planted):
    _, sig = planted
    seg = fmb_by_segment(sig["panel"])
    assert seg.loc["IG", "coef"] == pytest.approx(0.08, abs=0.03)
    assert seg.loc["IG", "t_stat"] > 3
    assert abs(seg.loc["HY", "t_stat"]) < 2           # nothing planted
    inter = fmb_interaction(sig["panel"])["summary"]
    assert inter.loc["spread_resid_IG", "coef"] == pytest.approx(0.08,
                                                                 abs=0.03)


def test_fmb_null_panel_finds_nothing():
    panel = synth_panel(T=40, n_bonds=200, mis_premium=0.0, seed=11)
    sig = spread_residuals(panel)
    seg = fmb_by_segment(sig["panel"])
    assert abs(seg.loc["IG", "t_stat"]) < 2.2


# ------------------------------------------------------------ portfolios
def test_quintile_sort_monotone_and_ls_positive(planted):
    _, sig = planted
    q = quintile_returns(sig["panel"], rating_class="0.IG", min_bonds=30)
    means = q.mean()
    assert means["Q4"] > means["Q0"]                  # cheap beats rich
    ls = long_short_stats(q)
    assert ls["t_stat"] > 2 and ls["ann_sharpe"] > 0.5


def test_factors_and_nw_regression(planted):
    _, sig = planted
    p = sig["panel"]
    fac = build_factors(p)
    assert list(fac.columns) == ["MKT", "CRD", "TERM"]
    assert len(fac) > 40
    q = quintile_returns(p, rating_class="0.IG", min_bonds=30)
    ls_ret = (q["Q4"] - q["Q0"]).dropna()
    reg = nw_regression(ls_ret, fac, lags=6)
    assert set(reg.index) == {"const", "MKT", "CRD", "TERM"}
    assert np.isfinite(reg["t_stat"]).all()


def test_two_pass_prices_mispricing(planted):
    _, sig = planted
    p = sig["panel"]
    fac = build_factors(p)
    out = two_pass_fmb(p, fac, min_months=24)
    s = out["summary"]
    assert s.loc["spread_resid_w", "t_stat"] > 2     # planted premium

# -------------------------------------------------------------- extensions
def test_decay_profile_grows_with_persistent_mispricing(planted):
    _, sig = planted
    dec = decay_profile(sig["panel"], horizons=(1, 3, 6))
    # h=1 slope = planted premium
    assert dec.loc[1, "coef_cum"] == pytest.approx(0.08, abs=0.03)
    # AR(1) errors (phi=0.7) keep paying: cumulative slope grows with h
    assert dec.loc[3, "coef_cum"] > dec.loc[1, "coef_cum"]
    assert dec.loc[6, "coef_cum"] > dec.loc[3, "coef_cum"]
    # ...at a decaying marginal rate
    assert dec.loc[6, "marginal"] < dec.loc[3, "marginal"]
    assert dec.loc[1, "t_nw"] > 3
    assert (dec["n_months"] > 40).all()


def test_liquidity_double_sort_runs_and_finds_premium_everywhere(planted):
    # premium planted independent of liquidity -> all buckets significant
    _, sig = planted
    liq = liquidity_double_sort(sig["panel"])
    assert {"low_liq", "high_liq"} <= set(liq.index)
    assert (liq["fmb_t"] > 2).all()


def test_strategy_turnover_bounds(planted):
    _, sig = planted
    t = strategy_turnover(sig["panel"])
    assert 0 < t["one_way_turnover"] <= 1
    assert t["mean_monthly_ls"] > 0
    assert t["breakeven_cost_bp"] > 0


def test_regime_conditional_fmb_shapes(planted):
    _, sig = planted
    out = regime_conditional_fmb(sig["panel"], jump_penalty=5.0)
    assert 0 <= out["stress_share"] <= 1
    assert len(out["table"]) >= 1
    assert out["table"]["coef"].notna().all()


# ------------------------------------------------------------- DD changes
def test_add_dd_changes_horizons():
    panel = synth_panel(T=30, n_bonds=50, seed=5)
    out = add_dd_changes(panel, horizons=(1, 3))
    assert {"DDCamp_chg1", "DDCamp_chg3"} <= set(out.columns)
    one = out[out["ISSUE_ID"] == 0].sort_values("DATE")
    manual = one["DDCamp"].diff(3)
    assert np.allclose(one["DDCamp_chg3"].iloc[3:], manual.iloc[3:],
                       atol=1e-12)
