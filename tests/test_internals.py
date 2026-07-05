"""Invariant and recovery tests for the toolkit's own machinery."""
import numpy as np
import pandas as pd
import pytest

from aim_toolkit import allocation as al
from aim_toolkit import stress, taa
from aim_toolkit.dashboard import Dashboard
from aim_toolkit.data import parse_maturity, simulate_market, simulate_yields
from aim_toolkit.fx import ECMFairValue, min_var_hedge_ratio
from aim_toolkit.regimes import GaussianMS, JumpModel
from aim_toolkit.yield_curve import DNSModel, ns_loadings

rng = np.random.default_rng(1)


# --------------------------------------------------------------- yield curve
def test_ns_loadings_limits():
    L = ns_loadings(np.array([1e-8, 1e4]), lam=0.5)
    assert L[0, 1] == pytest.approx(1.0, abs=1e-6)   # slope loading -> 1 at 0
    assert L[0, 2] == pytest.approx(0.0, abs=1e-6)   # curvature -> 0 at 0
    assert L[1, 1] == pytest.approx(0.0, abs=1e-3)   # slope -> 0 at infinity


def test_dns_fits_simulated_panel():
    model = DNSModel().fit(simulate_yields(n_months=200))
    assert model.rmse_bp < 5.0                       # measurement noise is 3bp
    fc = model.forecast_curve(h=6)
    assert fc.shape == (6, 11) and np.isfinite(fc.to_numpy()).all()


# ------------------------------------------------------------------- regimes
def test_gaussian_ms_recovers_regimes():
    mkt = simulate_market(n_days=1500)
    ms = GaussianMS().fit(mkt["equity_ret"].to_numpy())
    assert ms.sigma[0] < ms.sigma[1]                 # ordered: 0 = calm
    assert np.allclose(ms.P.sum(1), 1.0)
    states = (ms.smoothed[:, 1] > 0.5).astype(int)
    acc = max((states == mkt["true_state"]).mean(),
              (1 - states == mkt["true_state"]).mean())
    assert acc > 0.85


def test_jump_model_penalty_reduces_switching():
    mkt = simulate_market(n_days=1000)
    feat = mkt["equity_ret"].rolling(10).std().bfill().to_numpy()
    s_low = JumpModel(jump_penalty=1.0).fit(feat).states
    s_high = JumpModel(jump_penalty=200.0).fit(feat).states
    switches = lambda s: (np.diff(s) != 0).sum()
    assert switches(s_high) <= switches(s_low)


# ----------------------------------------------------------------------- taa
def test_purged_kfold_no_leakage():
    cv = taa.PurgedKFold(n_splits=5, label_horizon=21, embargo_pct=0.02)
    n = 1000
    for train, test in cv.split(n):
        t0, t1 = test[0], test[-1]
        assert not ((train >= t0 - 21) & (train <= t1 + 21)).any()  # purge
        assert not ((train > t1) & (train <= t1 + 21 + 20)).any()   # embargo


def test_backtest_no_lookahead():
    # position "knows" the big return at t=5; shift(1) must delay it to t=6
    ret = pd.Series(np.zeros(10)); ret.iloc[5] = 0.10
    pos = pd.Series(np.zeros(10)); pos.iloc[5] = 1.0
    res = taa.backtest(pos, ret, tcost_bp=0.0)
    assert res["gross"].iloc[5] == 0.0
    assert res["gross"].sum() == 0.0                 # missed the move entirely


def test_cv_sharpes_selection_is_out_of_sample():
    n = 800
    idx = pd.bdate_range("2020-01-01", periods=n)
    ret = pd.Series(rng.normal(5e-4, 0.01, n), index=idx)
    prices = (1 + ret).cumprod() * 100
    fn = lambda inputs, lookback: taa.zscore_position(
        taa.momentum(inputs["p"], lookback=lookback, skip=5)).fillna(0)
    cv = taa.PurgedKFold(n_splits=4, label_horizon=10)
    table, sel = taa.cv_sharpes(fn, [{"lookback": k} for k in (60, 120)],
                                {"p": prices}, ret, cv)
    assert len(table) == 2 and len(sel) == 4
    assert {"train_sharpe", "oos_sharpe"} <= set(sel.columns)


def test_psr_sanity():
    good = pd.Series(rng.normal(0.001, 0.01, 1000))
    flat = pd.Series(rng.normal(0.0, 0.01, 1000))
    assert taa.probabilistic_sharpe(good) > taa.probabilistic_sharpe(flat)
    assert 0.0 < taa.probabilistic_sharpe(flat) < 1.0
    assert taa.deflated_sharpe(good, n_trials=10) \
        <= taa.probabilistic_sharpe(good)


# -------------------------------------------------------------------- stress
def _flat_curve(t):
    return np.full_like(np.asarray(t, float), 0.03)


def _demo_portfolio():
    return stress.Portfolio(mv=10_000.0,
                            krd={2: 0.4, 5: 1.2, 10: 3.0, 20: 2.4, 30: 1.0},
                            spread_dur=5.5, credit_weight=0.45,
                            equity_weight=0.08, fx_unhedged_weight=0.05)


def test_parallel_shock_equals_total_duration():
    pf = _demo_portfolio()
    pnl = stress.asset_pnl(pf, {"rates": {k: 100 for k in pf.krd}})
    assert pnl["rates"] == pytest.approx(-sum(pf.krd.values()) / 100 * pf.mv)


def test_liability_krds_sum_to_duration():
    liab = stress.LiabilityBook(cashflows=np.full(30, 100.0))
    _, dur = liab.pv_and_duration(_flat_curve)
    krds = stress.liability_krd(liab, _flat_curve, [2, 5, 10, 20, 30])
    assert sum(krds.values()) == pytest.approx(dur, rel=1e-3)


def test_krd_gap_consistent_with_parallel_gap():
    pf = _demo_portfolio()
    liab = stress.LiabilityBook(cashflows=np.full(30, 100.0))
    gap_tbl = stress.krd_gap(pf, liab, _flat_curve)
    parallel = stress.duration_gap(pf, liab, _flat_curve,
                                   asset_dur=sum(pf.krd.values()))
    assert gap_tbl["surplus_chg_+100bp"].sum() == pytest.approx(
        parallel["surplus_chg_+100bp"], abs=1.0)


# ------------------------------------------------------------------------ fx
def test_parse_maturity():
    assert parse_maturity("3M") == pytest.approx(0.25)
    assert parse_maturity("10Y") == pytest.approx(10.0)
    assert parse_maturity("0.5") == 0.5
    assert parse_maturity(7) == 7.0
    with pytest.raises(ValueError):
        parse_maturity("banana")


def test_ecm_detects_cointegration():
    n = 600
    fund = np.cumsum(rng.normal(0, 0.01, n))          # random-walk fundamental
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + rng.normal(0, 0.01)   # stationary gap
    spot = pd.Series(0.5 + 1.0 * fund + u)
    m = ECMFairValue().fit(spot, pd.DataFrame({"f": fund}))
    assert m.cointegrated_5pct
    assert m.gamma < 0 and np.isfinite(m.half_life)
    assert m.beta[1] == pytest.approx(1.0, abs=0.15)


def test_min_var_hedge_ratio_recovers_beta():
    n = 600
    fx = pd.Series(rng.normal(0, 0.006, n))
    asset = 0.8 * fx + pd.Series(rng.normal(0, 0.001, n))
    h = min_var_hedge_ratio(asset, fx, window=250).dropna()
    assert h.mean() == pytest.approx(0.8, abs=0.05)


# -------------------------------------------------------------- allocation
def test_black_litterman_tight_views_pull_to_q():
    Sigma = np.diag([0.04, 0.02, 0.01])
    w = np.array([0.5, 0.3, 0.2])
    P, Q = np.eye(3), np.array([0.05, 0.03, 0.01])
    mu, cov = al.black_litterman(Sigma, w, P, Q, omega=np.eye(3) * 1e-12)
    assert np.allclose(mu, Q, atol=1e-4)
    # posterior predictive covariance must dominate Sigma (PSD order)
    assert np.all(np.linalg.eigvalsh(cov - Sigma) >= -1e-12)


def test_entropy_pooling_hits_view_and_is_valid():
    scen = rng.normal(0, 0.05, (2000, 3))
    target = scen[:, 0].mean() + 0.02
    ep = al.EntropyPooling().fit(scen, al.view_on_mean(scen, 0), [target])
    mu, _ = ep.posterior_moments()
    assert mu[0] == pytest.approx(target, abs=1e-4)
    assert ep.q.min() >= 0 and ep.q.sum() == pytest.approx(1.0)
    assert ep.kl >= 0 and 1 <= ep.effective_n <= len(scen)


def test_mv_optimize_respects_constraints():
    mu = np.array([0.05, 0.03, 0.02])
    Sigma = np.diag([0.05, 0.02, 0.01])
    w = al.mv_optimize(mu, Sigma, w_max=0.5)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all() and (w <= 0.5 + 1e-9).all()


# --------------------------------------------------------------- dashboard
def test_dashboard_writes_utf8(tmp_path):
    d = Dashboard("Test ±β→", "±2σ bands")
    d.add_section("Ω section", note="±β→ non-ASCII")
    out = d.save(str(tmp_path / "x.html"))
    html = (tmp_path / "x.html").read_text(encoding="utf-8")
    assert "±β→" in html and "Ω" in html
