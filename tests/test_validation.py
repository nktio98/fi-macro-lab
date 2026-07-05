"""Validate from-scratch estimators against statsmodels reference
implementations. This is the proof that the hand-rolled econometrics
(ADF, Newey-West, BH-FDR, VAR) are correct, not just plausible."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from aim_toolkit.fx import adf_tstat
from aim_toolkit.managers import _newey_west_se, benjamini_hochberg, factor_regression
from aim_toolkit.yield_curve import VAR1

rng = np.random.default_rng(0)


def test_adf_tstat_matches_statsmodels():
    # stationary AR(1) residual-like series
    u = np.zeros(400)
    for t in range(1, 400):
        u[t] = 0.7 * u[t - 1] + rng.normal()
    ours = adf_tstat(u, lags=1)
    ref = adfuller(u, maxlag=1, regression="n", autolag=None)[0]
    assert ours == pytest.approx(ref, abs=1e-8)


def test_newey_west_matches_statsmodels():
    T = 300
    X = np.column_stack([np.ones(T), rng.normal(size=(T, 2))])
    # autocorrelated errors so HAC actually matters
    e = np.zeros(T)
    for t in range(1, T):
        e[t] = 0.5 * e[t - 1] + rng.normal()
    y = X @ np.array([0.1, 1.0, -0.5]) + e
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ours = _newey_west_se(X, resid, lags=6)
    ref = sm.OLS(y, X).fit(cov_type="HAC",
                           cov_kwds={"maxlags": 6, "use_correction": False})
    assert np.allclose(ours, ref.bse, atol=1e-10)


def test_benjamini_hochberg_matches_statsmodels():
    p = pd.Series(rng.uniform(0, 0.2, 25),
                  index=[f"mgr_{i}" for i in range(25)])
    ours = benjamini_hochberg(p, fdr=0.10)["significant_at_FDR"]
    ref = multipletests(p.to_numpy(), alpha=0.10, method="fdr_bh")[0]
    assert (ours.to_numpy() == ref).all()


def test_var1_matches_statsmodels():
    T = 500
    Z = np.zeros((T, 2))
    A_true = np.array([[0.8, 0.1], [0.0, 0.7]])
    for t in range(1, T):
        Z[t] = np.array([0.2, -0.1]) + A_true @ Z[t - 1] + rng.normal(0, 0.1, 2)
    F = pd.DataFrame(Z, columns=["a", "b"])
    ours = VAR1().fit(F)
    ref = VAR(Z).fit(1, trend="c")
    assert np.allclose(ours.c, ref.params[0], atol=1e-8)
    assert np.allclose(ours.A, ref.params[1:].T, atol=1e-8)


def test_factor_regression_betas_match_ols():
    T = 240
    factors = pd.DataFrame(rng.normal(0, 0.03, (T, 2)), columns=["mkt", "val"])
    y = 0.001 + 0.9 * factors["mkt"] + 0.3 * factors["val"] \
        + rng.normal(0, 0.01, T)
    out = factor_regression(pd.Series(y), factors, nw_lags=6)
    X = sm.add_constant(factors.to_numpy())
    ref = sm.OLS(y.to_numpy(), X).fit()
    assert out["beta_mkt"] == pytest.approx(ref.params[1], abs=1e-10)
    assert out["beta_val"] == pytest.approx(ref.params[2], abs=1e-10)
    assert out["alpha_ann_%"] == pytest.approx(ref.params[0] * 12 * 100, abs=1e-8)
