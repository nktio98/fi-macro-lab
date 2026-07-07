"""Tests for the advanced additions: local projections, nowcasting,
AFNS/ACM, GARCH-DCC, Monte Carlo stress, and signal monitoring."""
import numpy as np
import pandas as pd
import pytest

from aim_toolkit import monitoring, stress
from aim_toolkit.data import simulate_yields
from aim_toolkit.fx import dcc_hedge_ratio, garch11
from aim_toolkit.macro import irf_summary, local_projections
from aim_toolkit.nowcast import activity_factor
from aim_toolkit.yield_curve import ACMTermPremium, AFNSModel, DNSModel, \
    afns_adjustment

rng = np.random.default_rng(2)


# ------------------------------------------------------ local projections
def test_lp_recovers_contemporaneous_effect():
    n = 800
    shock = pd.Series(rng.normal(0, 1, n))
    y = pd.Series(0.5 * shock + rng.normal(0, 0.3, n))
    irf = local_projections(y, shock, horizons=6, lags=2)
    assert irf.loc[0, "beta"] == pytest.approx(0.5, abs=0.06)
    # no propagation in the DGP -> cumulative IRF stays flat at ~0.5
    assert irf.loc[6, "beta"] == pytest.approx(0.5, abs=0.15)
    assert (irf["hi"] >= irf["beta"]).all() and (irf["lo"] <= irf["beta"]).all()


def test_lp_flags_significance_correctly():
    n = 600
    shock = pd.Series(rng.normal(0, 1, n))
    noise = pd.Series(rng.normal(0, 1, n))
    irf_null = local_projections(noise, shock, horizons=4)
    assert not irf_summary(irf_null)["significant"]
    irf_real = local_projections(
        pd.Series(0.8 * shock + rng.normal(0, 0.2, n)), shock, horizons=4)
    assert irf_summary(irf_real)["significant"]


# ------------------------------------------------------------- nowcasting
def test_activity_factor_recovers_common_factor_with_missing():
    T, N = 200, 5
    f_true = np.cumsum(rng.normal(0, 1, T)) / 10
    panel = pd.DataFrame(
        {f"x{i}": 0.8 * f_true + rng.normal(0, 0.4, T) for i in range(N)},
        index=pd.date_range("2005-01-31", periods=T, freq="ME"))
    panel.iloc[-3:, 0] = np.nan            # ragged edge
    panel.iloc[10:20, 2] = np.nan          # interior gap
    factor, loadings = activity_factor(panel)
    corr = np.corrcoef(factor.to_numpy(), f_true)[0, 1]
    assert abs(corr) > 0.9
    assert (loadings > 0).all()            # sign-aligned to activity
    assert factor.std() == pytest.approx(1.0, abs=0.02)   # ddof convention


# --------------------------------------------------------------- AFNS/ACM
def test_afns_adjustment_properties():
    taus = np.array([1.0, 5.0, 10.0, 30.0])
    zero = afns_adjustment(taus, 0.5, np.zeros(3))
    assert np.allclose(zero, 0)
    adj = afns_adjustment(taus, 0.5, np.array([0.01, 0.01, 0.01]))
    assert (adj > 0).all()
    assert adj[-1] > adj[0]               # convexity grows with maturity


def test_afns_lowers_long_yields_vs_dns():
    yields = simulate_yields(n_months=240)
    dns = DNSModel().fit(yields)
    afns = AFNSModel().fit(yields)
    y_dns = dns.reconstruct(dns.factors).iloc[-1]
    y_afns = afns.reconstruct(afns.factors).iloc[-1]
    assert y_afns[30.0] < y_dns[30.0]     # long end pulled down
    assert abs(y_afns[0.25] - y_dns[0.25]) < 0.02   # short end ~unchanged
    assert np.isfinite(afns.rmse_bp)


def test_acm_decomposition_identity_and_fit():
    yields = simulate_yields(n_months=300)
    acm = ACMTermPremium().fit(yields, k_factors=3, max_years=10)
    fitted = acm.fitted_yield(10)
    rn = acm.risk_neutral_yield(10)
    tp = acm.term_premium(10)
    assert np.allclose(fitted - rn, tp)   # identity holds exactly
    assert acm.fit_rmse_bp(10) < 50       # model prices the 10y decently
    assert acm.fit_rmse_bp(5) < 50


# -------------------------------------------------------------- GARCH/DCC
def test_garch11_recovers_persistence():
    T = 4000
    omega, alpha, beta = 0.02, 0.08, 0.90
    r = np.zeros(T); h = omega / (1 - alpha - beta)
    for t in range(1, T):
        h = omega + alpha * r[t - 1] ** 2 + beta * h
        r[t] = np.sqrt(h) * rng.normal()
    fit = garch11(r)
    assert 0.90 < fit["alpha"] + fit["beta"] < 1.0
    assert (fit["cond_vol"] > 0).all()


def test_dcc_recovers_constant_correlation():
    T = 2000
    rho = 0.6
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], T) * 0.01
    out = dcc_hedge_ratio(pd.Series(z[:, 0]), pd.Series(z[:, 1]))
    assert out["rho"].iloc[100:].mean() == pytest.approx(rho, abs=0.12)
    # constant-rho data is degenerate for DCC: a -> 0, b unidentified,
    # so only check the parameters stayed inside their bounds
    assert 0 < out["a"] < 0.3 and 0.5 <= out["b"] < 0.999
    assert np.isfinite(out["hedge_ratio"]).all()


# ------------------------------------------------------------ Monte Carlo
def _pf():
    return stress.Portfolio(mv=10_000.0, krd={5: 3.0, 10: 3.0},
                            spread_dur=5.0, credit_weight=0.4,
                            equity_weight=0.1, fx_unhedged_weight=0.05)


def test_monte_carlo_degenerate_matches_deterministic():
    row = {"rates_bp": 50.0, "spreads_bp": 20.0,
           "equity_pct": -5.0, "fx_pct": -2.0}
    moves = pd.DataFrame([row] * 30)
    out = stress.monte_carlo_pnl(_pf(), moves, n_sims=200)
    det = stress.asset_pnl(_pf(), {"rates": {5: 50.0, 10: 50.0},
                                   "spreads_bp": 20.0, "equity_pct": -5.0,
                                   "fx_pct": -2.0})["total"]
    assert np.allclose(out["pnl"], det)


def test_monte_carlo_var_ordering():
    moves = pd.DataFrame({
        "rates_bp": rng.normal(0, 30, 300),
        "spreads_bp": rng.normal(0, 20, 300),
        "equity_pct": rng.normal(0, 4, 300),
        "fx_pct": rng.normal(0, 2, 300)})
    for method in ("bootstrap", "normal"):
        out = stress.monte_carlo_pnl(_pf(), moves, n_sims=5000,
                                     method=method)
        assert out["var99"] >= out["var95"] > 0
        assert out["es95"] >= out["var95"]
        assert out["es99"] >= out["var99"]


# -------------------------------------------------------------- monitoring
def test_forecast_eval_perfect_and_noise():
    r = pd.Series(rng.normal(0, 1, 300))
    perfect = monitoring.forecast_eval(r, r)
    assert perfect["hit_rate"] == 1.0 and perfect["rmse"] == 0.0
    noise = monitoring.forecast_eval(pd.Series(rng.normal(0, 1, 300)), r)
    assert 0.35 < noise["hit_rate"] < 0.65


def test_cusum_flags_drift_not_stationary():
    idx = pd.RangeIndex(400)
    stationary = pd.Series(rng.normal(0, 1, 400), index=idx)
    assert not monitoring.cusum_drift(stationary).iloc[-1]["flag"]
    shifted = stationary.copy()
    shifted.iloc[200:] += 2.0              # structural break
    assert monitoring.cusum_drift(shifted).iloc[-1]["flag"]


def test_decay_report_stable_signal():
    good = pd.Series(rng.normal(8e-4, 0.01, 1500))
    rep = monitoring.decay_report(good)
    assert rep["verdict"] in ("stable", "degrading")
    assert rep["full_ir"] > 0.5
