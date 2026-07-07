"""Tests for Kalman-MLE DNS, shadow-rate UKF, BVAR, Smith-Wilson, and
the Fama-French luck-vs-skill bootstrap."""
import numpy as np
import pandas as pd
import pytest

from aim_toolkit import stress
from aim_toolkit.bvar import MinnesotaBVAR
from aim_toolkit.data import simulate_yields
from aim_toolkit.managers import bootstrap_skill_test
from aim_toolkit.statespace import KalmanDNS, ShadowRateDNS
from aim_toolkit.yield_curve import DNSModel, ns_loadings, smith_wilson

rng = np.random.default_rng(3)


# ------------------------------------------------------------- Kalman MLE
def test_kalman_dns_improves_on_two_step():
    yields = simulate_yields(n_months=150)      # DGP matches the model
    two = DNSModel().fit(yields)
    km = KalmanDNS(maxiter=40).fit(yields)
    assert km.loglik >= km.loglik_init - 1e-6   # MLE never worse than init
    assert 0.3 < km.lam < 0.9                   # true lambda = 0.55
    assert km.rmse_bp < two.rmse_bp * 1.2
    # smoothed factors track the two-step factors closely
    corr = km.factors["level"].corr(two.factors["level"])
    assert corr > 0.95
    fc = km.forecast_curve(6)
    assert fc.shape == (6, yields.shape[1])
    assert np.isfinite(fc.to_numpy()).all()


# ------------------------------------------------------------ shadow rate
def _censored_panel(T=200, lam=0.5):
    """Yield panel where the TRUE shadow short rate dips below zero."""
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    level = np.full(T, 1.6)
    slope = np.concatenate([np.linspace(-0.5, -3.0, T // 2),
                            np.linspace(-3.0, -0.5, T - T // 2)])
    curv = np.full(T, -0.5)
    F = np.column_stack([level, slope, curv])
    F += rng.normal(0, 0.05, F.shape)
    shadow = F @ ns_loadings(mats, lam).T
    obs = np.maximum(shadow, 0.0) + rng.normal(0, 0.02, shadow.shape)
    idx = pd.date_range("2008-01-31", periods=T, freq="ME")
    Y = pd.DataFrame(np.maximum(obs, 0.0), index=idx, columns=mats)
    true_short = pd.Series(shadow[:, 0], index=idx)
    return Y, true_short


def test_shadow_rate_goes_below_bound():
    Y, true_short = _censored_panel()
    sr = ShadowRateDNS(lb=0.0).fit(Y)
    est = sr.shadow_short_rate(0.25)
    censored = true_short < -0.2                # deep-ZLB stretch
    assert censored.sum() > 30
    # shadow estimate is negative where truth is negative
    assert (est[censored] < 0).mean() > 0.7
    # and tracks the true shadow rate's shape
    assert est[censored].corr(true_short[censored]) > 0.6
    # observed reconstruction stays decent
    assert sr.rmse_bp < 25


# ------------------------------------------------------------------ BVAR
def _sim_var1(T=400, k=3):
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    Z = np.zeros((T, k))
    for t in range(1, T):
        Z[t] = A @ Z[t - 1] + rng.normal(0, 1.0, k)
    return pd.DataFrame(Z, columns=["a", "b", "c"]), A


def test_bvar_loose_prior_recovers_var():
    Y, A_true = _sim_var1()
    bv = MinnesotaBVAR(lags=1, lambda1=5.0).fit(Y)
    A_post = bv.B[1:].T
    assert np.abs(A_post - A_true).max() < 0.15


def test_bvar_tight_prior_shrinks_to_prior_mean():
    Y, _ = _sim_var1()
    bv = MinnesotaBVAR(lags=1, lambda1=0.01, own_mean=0.0).fit(Y)
    assert np.abs(bv.B[1:]).max() < 0.05        # everything shrunk to ~0


def test_bvar_simulation_moments_and_stress_plumbing():
    Y, _ = _sim_var1()
    Y.columns = ["rates_bp", "spreads_bp", "equity_pct"]
    Y["fx_pct"] = rng.normal(0, 1, len(Y))
    bv = MinnesotaBVAR(lags=1, lambda1=0.2).fit(Y)
    sims = bv.simulate(h=1, n_draws=2000, seed=1)
    assert sims.shape == (2000, 4)
    assert 0.5 < sims["rates_bp"].std() / Y["rates_bp"].std() < 2.0
    pf = stress.Portfolio(mv=10_000.0, krd={5: 3.0, 10: 3.0},
                          spread_dur=5.0, credit_weight=0.4,
                          equity_weight=0.1, fx_unhedged_weight=0.05)
    out = stress.monte_carlo_pnl(pf, sims, method="given")
    assert len(out["pnl"]) == 2000
    assert out["var99"] >= out["var95"]


# ---------------------------------------------------------- Smith-Wilson
def test_smith_wilson_reprices_and_converges_to_ufr():
    tenors = np.array([1.0, 2, 5, 10, 20, 30])
    zeros = np.array([2.0, 2.2, 2.6, 3.0, 3.2, 3.1])
    fn = smith_wilson(tenors, zeros, ufr=0.038, alpha=0.15)
    for t, z in zip(tenors, zeros):
        assert fn(t) == pytest.approx(z / 100, abs=1e-8)   # exact repricing
    f_inf = np.log(1.038)
    fwd = (fn(100.0) * 100 - fn(90.0) * 90) / 10           # 90->100y forward
    assert fwd == pytest.approx(f_inf, abs=0.002)
    v = fn(np.array([15.0, 40.0]))
    assert v.shape == (2,) and np.all(np.isfinite(v))


# ---------------------------------------------------- luck-vs-skill boot
def _manager_panel(alpha_star=0.0, M=15, T=120):
    fac = pd.DataFrame(rng.normal(0.004, 0.03, (T, 2)),
                       columns=["mkt", "credit"])
    rets = {}
    for i in range(M):
        a = alpha_star if i == 0 else 0.0
        rets[f"m{i:02d}"] = a + 0.9 * fac["mkt"] + 0.2 * fac["credit"] \
            + rng.normal(0, 0.01, T)
    return pd.DataFrame(rets), fac


def test_bootstrap_skill_null_panel_not_flagged():
    R, fac = _manager_panel(alpha_star=0.0)
    out = bootstrap_skill_test(R, fac, n_boot=300, seed=4)
    assert out["p_top"] > 0.05                  # luck explains the best t


def test_bootstrap_skill_detects_true_alpha():
    R, fac = _manager_panel(alpha_star=0.005)   # ~50bp/month true alpha
    out = bootstrap_skill_test(R, fac, n_boot=300, seed=4)
    assert out["p_top"] < 0.05
    assert out["actual_sorted_t"][-1] > out["null_95"][-1]
