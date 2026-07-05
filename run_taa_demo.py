"""TAA demo: purged CV over a signal grid + deflated Sharpe verdict."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aim_toolkit import taa

OUT = "outputs"
rng = np.random.default_rng(3)


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# Simulate an asset with a WEAK real momentum effect (so the honest answer
# is "marginal", which is the realistic case) - plus pure-noise features.
n = 3000
true_signal = np.zeros(n)
for t in range(1, n):
    true_signal[t] = 0.98 * true_signal[t - 1] + rng.normal(0, 0.15)
ret = 0.00035 + 0.0006 * np.tanh(true_signal) + rng.normal(0, 0.009, n)
idx = pd.bdate_range("2014-01-01", periods=n)
asset_ret = pd.Series(ret, index=idx)
prices = (1 + asset_ret).cumprod() * 100

section("1. SIGNAL GRID + PURGED/EMBARGOED CROSS-VALIDATION")
def make_position(inputs, lookback, skip):
    sig = taa.momentum(inputs["prices"], lookback=lookback, skip=skip)
    return taa.zscore_position(sig).fillna(0.0)

grid = [{"lookback": lb, "skip": sk}
        for lb in (63, 126, 252) for sk in (5, 21)]
cv = taa.PurgedKFold(n_splits=5, label_horizon=21, embargo_pct=0.02)
res, selection = taa.cv_sharpes(make_position, grid, {"prices": prices},
                                asset_ret, cv)
res = res.sort_values("test_sharpe_mean", ascending=False)
print(res.to_string(index=False))
print("\nTrain-selected config per fold and its OUT-OF-SAMPLE Sharpe:")
print(selection.to_string())
print(f"\nOOS Sharpe of the selection procedure: "
      f"{selection['oos_sharpe'].mean():.2f} "
      "(this is the honest number - selection never sees its test fold)")

section("2. DEFLATED SHARPE - correcting for the number of trials")
best = res.iloc[0]
pos = make_position({"prices": prices}, int(best["lookback"]), int(best["skip"]))
net = taa.backtest(pos, asset_ret)["net"]
sr = taa.sharpe(net)
psr = taa.probabilistic_sharpe(net)
# cross-trial variance of the grid's full-sample Sharpes (per-period units)
grid_srs = np.array([taa.sharpe(taa.backtest(
    make_position({"prices": prices}, g["lookback"], g["skip"]),
    asset_ret)["net"]) for g in grid]) / np.sqrt(taa.ANN)
dsr = taa.deflated_sharpe(net, n_trials=len(grid),
                          trial_sr_var=float(grid_srs.var()))
print(f"Best config: lookback={int(best['lookback'])}, skip={int(best['skip'])}")
print(f"Full-sample net Sharpe : {sr:.2f}")
print(f"Probabilistic Sharpe   : {psr:.1%}  (P(true SR > 0), one strategy)")
print(f"Deflated Sharpe        : {dsr:.1%}  (corrected for {len(grid)} trials)")
verdict = "DEPLOYABLE (survives multiple-testing correction)" if dsr > 0.95 \
    else "NOT PROVEN - likely selection bias; do not deploy on this evidence"
print(f"Verdict: {verdict}")

# noise benchmark: best of 6 random signals often 'looks' good pre-deflation
noise_srs = []
for _ in range(len(grid)):
    npos = taa.zscore_position(pd.Series(
        rng.normal(0, 1, n), index=idx).rolling(20).mean()).fillna(0)
    noise_srs.append(taa.sharpe(taa.backtest(npos, asset_ret)["net"]))
print(f"\nFor context: best of {len(grid)} PURE-NOISE strategies scored "
      f"Sharpe {max(noise_srs):.2f}\n(this is what naive backtest selection "
      "reports as 'skill').")

fig, ax = plt.subplots(figsize=(10, 4))
cum = (1 + net).cumprod()
ax.plot(cum.index, cum, label=f"best momentum config (net SR {sr:.2f})")
ax.plot(cum.index, (1 + asset_ret).cumprod() / 100 * cum.iloc[0] * 100 /
        (1 + asset_ret).cumprod().iloc[0], alpha=0)  # keep scale simple
ax.set_title("TAA strategy equity curve (net of costs)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/7_taa.png", dpi=130)
print("\nDone.")
