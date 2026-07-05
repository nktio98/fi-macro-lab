"""
Tactical asset allocation: signals + honest backtesting.

The scientific core here is NOT the signals (those are simple, deliberately)
but the validation machinery that separates professional quant work from
curve-fitting:

  - PurgedKFold  : time-series cross-validation with purging (drop training
    samples whose label window overlaps the test set) and an embargo buffer
    after each test fold (Lopez de Prado, "Advances in Financial ML").
  - probabilistic_sharpe / deflated_sharpe : PSR corrects the Sharpe ratio
    for non-normality and sample length; DSR additionally corrects for the
    number of strategies tried (multiple testing) using the expected max
    Sharpe under the null (Bailey & Lopez de Prado 2014).

A strategy is only interesting if its DEFLATED Sharpe is significant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = 252


# ------------------------------------------------------------- signals
def momentum(prices: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Classic 12-1 momentum: return over lookback excluding last `skip` days."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def value_z(yield_series: pd.Series, window: int = 756) -> pd.Series:
    """Valuation z-score: current yield vs trailing distribution (high = cheap)."""
    mu = yield_series.rolling(window).mean()
    sd = yield_series.rolling(window).std()
    return (yield_series - mu) / sd


def carry(yield_series: pd.Series, funding: pd.Series) -> pd.Series:
    return yield_series - funding


def zscore_position(signal: pd.Series, window: int = 252,
                    cap: float = 2.0) -> pd.Series:
    """Convert signal to position in [-cap, cap] via rolling z-score."""
    z = (signal - signal.rolling(window).mean()) / signal.rolling(window).std()
    return z.clip(-cap, cap)


# --------------------------------------------------------- purged CV
class PurgedKFold:
    """K-fold CV for financial series: purge overlapping labels + embargo."""

    def __init__(self, n_splits: int = 5, label_horizon: int = 21,
                 embargo_pct: float = 0.02):
        self.k, self.h, self.embargo_pct = n_splits, label_horizon, embargo_pct

    def split(self, n: int):
        idx = np.arange(n)
        embargo = int(n * self.embargo_pct)
        folds = np.array_split(idx, self.k)
        for test in folds:
            t0, t1 = test[0], test[-1]
            train = idx[(idx < t0 - self.h) |            # purge pre-overlap
                        (idx > t1 + self.h + embargo)]   # purge + embargo post
            yield train, test


# ---------------------------------------------------------- backtest
def backtest(position: pd.Series, asset_ret: pd.Series,
             tcost_bp: float = 2.0) -> pd.DataFrame:
    """Daily P&L of position (entered at close, earns next-day return),
    net of proportional transaction costs on position changes."""
    pos = position.shift(1).fillna(0.0)
    gross = pos * asset_ret
    costs = pos.diff().abs().fillna(0.0) * tcost_bp / 1e4
    net = gross - costs
    return pd.DataFrame({"gross": gross, "costs": costs, "net": net})


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: P(true SR > benchmark), adjusting for skew/kurtosis/sample size."""
    r = returns.dropna()
    T = len(r)
    sr = r.mean() / r.std()                      # per-period SR
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3          # raw kurtosis
    sr_b = sr_benchmark / np.sqrt(ANN)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_b) * np.sqrt(T - 1) / denom))


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    trial_sr_var: float | None = None) -> float:
    """DSR: PSR against the expected max Sharpe from n_trials tries.

    trial_sr_var should be the CROSS-TRIAL variance of the tried strategies'
    Sharpe estimates (Bailey & Lopez de Prado 2014). The default 1/T is the
    sampling variance of a single SR estimate under the null -- a floor, so
    the default DSR is conservative-optimistic; pass the true cross-trial
    variance when the whole grid's Sharpes are available."""
    r = returns.dropna()
    sr_var = trial_sr_var if trial_sr_var is not None else (1.0 / len(r))
    em = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_max = np.sqrt(sr_var) * ((1 - em) * z1 + em * z2)   # per-period
    return probabilistic_sharpe(r, sr_benchmark=sr_max * np.sqrt(ANN))


def cv_sharpes(position_fn, params_grid: list[dict], signal_inputs,
               asset_ret: pd.Series, cv: PurgedKFold
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purged CV with train-based selection.

    For each fold, every candidate parameter set is scored on the TRAINING
    indices only; the train-best candidate is then evaluated once on the
    held-out TEST fold. position_fn(inputs, **params) must return a causal
    (backward-looking) position series aligned to asset_ret.

    Returns (table, selection):
      table     -- per-candidate train/test Sharpe summary across folds.
      selection -- per fold: the train-selected params and their out-of-
                   sample (test) Sharpe. selection["oos_sharpe"].mean() is
                   the OOS estimate of the *selection procedure* -- the
                   number to trust, since it never sees its own test data.

    Train indices can be non-contiguous (blocks before and after the test
    fold); the single misaligned observation at the seam is negligible at
    daily frequency.
    """
    n = len(asset_ret)
    positions = [position_fn(signal_inputs, **p) for p in params_grid]
    folds = list(cv.split(n))
    train_sr = np.zeros((len(folds), len(params_grid)))
    test_sr = np.zeros((len(folds), len(params_grid)))
    for f, (train, test) in enumerate(folds):
        for j, pos in enumerate(positions):
            train_sr[f, j] = sharpe(
                backtest(pos.iloc[train], asset_ret.iloc[train])["net"])
            test_sr[f, j] = sharpe(
                backtest(pos.iloc[test], asset_ret.iloc[test])["net"])
    table = pd.DataFrame(
        [{**params_grid[j],
          "train_sharpe_mean": train_sr[:, j].mean(),
          "test_sharpe_mean": test_sr[:, j].mean(),
          "test_sharpe_std": test_sr[:, j].std()}
         for j in range(len(params_grid))]).round(3)
    picks = train_sr.argmax(axis=1)
    selection = pd.DataFrame(
        [{**params_grid[j],
          "train_sharpe": train_sr[f, j], "oos_sharpe": test_sr[f, j]}
         for f, j in enumerate(picks)],
        index=pd.RangeIndex(len(folds), name="fold")).round(3)
    return table, selection
