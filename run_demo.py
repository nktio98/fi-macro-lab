"""End-to-end demo: yield curve engine -> regimes -> ALM stress test."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aim_toolkit.data import simulate_yields, simulate_market, MATURITIES
from aim_toolkit.yield_curve import DNSModel, ns_loadings
from aim_toolkit.regimes import GaussianMS, JumpModel, regime_summary
from aim_toolkit import stress

OUT = "outputs"


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---------------------------------------------------------------- 1. DNS
section("1. DYNAMIC NELSON-SIEGEL YIELD CURVE ENGINE")
yields = simulate_yields()
model = DNSModel().fit(yields)
print(f"Optimal lambda: {model.lam:.3f}   |   in-sample RMSE: {model.rmse_bp:.1f} bp")
print("\nLatest factors (level / slope / curvature, %):")
print(model.factors.iloc[-1].round(3).to_string())
print("\nVAR(1) long-run factor means (%):",
      np.round(model.var.long_run_mean(), 2))

fc = model.forecast_curve(h=12)
last_curve = yields.iloc[-1]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(model.factors.index, model.factors["level"], label="level")
ax[0].plot(model.factors.index, model.factors["slope"], label="slope")
ax[0].plot(model.factors.index, model.factors["curvature"], label="curvature")
ax[0].set_title("DNS factors (extracted)"); ax[0].legend(); ax[0].set_ylabel("%")
ax[1].plot(MATURITIES, last_curve, "o-", label="current curve")
ax[1].plot(MATURITIES, fc.iloc[2], "s--", label="forecast +3m")
ax[1].plot(MATURITIES, fc.iloc[-1], "^--", label="forecast +12m")
ax[1].set_title("Curve forecast (VAR(1) on factors)")
ax[1].set_xlabel("maturity (yrs)"); ax[1].set_ylabel("%"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/1_yield_curve.png", dpi=130)

# ------------------------------------------------------------- 2. Regimes
section("2. REGIME DETECTION (Markov-switching vs statistical jump model)")
mkt = simulate_market()
ret = mkt["equity_ret"]

ms = GaussianMS().fit(ret.to_numpy())
print("Markov-switching: state vols (daily): "
      f"{np.round(ms.sigma, 4)}  | expected duration (days): "
      f"{np.round(ms.expected_duration, 1)}")
ms_states = (ms.smoothed[:, 1] > 0.5).astype(int)

feat = np.column_stack([
    ret.rolling(10).std().bfill(),
    ret.rolling(10).mean().bfill(),
    mkt["credit_spread_bp"].diff().rolling(10).mean().bfill(),
])
jm = JumpModel(jump_penalty=80.0).fit(feat)

acc_ms = max((ms_states == mkt["true_state"]).mean(),
             (1 - ms_states == mkt["true_state"]).mean())
acc_jm = max((jm.states == mkt["true_state"]).mean(),
             (1 - jm.states == mkt["true_state"]).mean())
print(f"\nAccuracy vs true simulated regime:  MS-HMM {acc_ms:.1%}   "
      f"JumpModel {acc_jm:.1%}")
print("\nPer-regime stats (Jump model):")
print(regime_summary(jm.states, ret).to_string())
switches = lambda s: int((np.diff(s) != 0).sum())
print(f"\nRegime switches:  MS-HMM {switches(ms_states)}  |  "
      f"JumpModel {switches(jm.states)}  (fewer = more tradable)")

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
cum = (1 + ret).cumprod()
for a, states, name in [(ax[0], ms_states, "Markov-switching"),
                        (ax[1], jm.states, "Statistical jump model")]:
    a.plot(mkt.index, cum, color="k", lw=0.8)
    a.fill_between(mkt.index, cum.min(), cum.max(), where=states == 1,
                   alpha=0.25, color="red", label="stress regime")
    a.set_title(f"{name}: detected stress regimes"); a.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/2_regimes.png", dpi=130)

# ------------------------------------------------- 3. ALM + stress engine
section("3. INSURANCE ALM STRESS TEST (stylized Asian life portfolio)")
pf = stress.Portfolio(
    mv=10_000.0,                                # EUR mn
    krd={2: 0.4, 5: 1.2, 10: 3.0, 20: 2.4, 30: 1.0},   # sums to 8.0y
    spread_dur=5.5, credit_weight=0.45,
    equity_weight=0.08, fx_unhedged_weight=0.05)
asset_dur = sum(pf.krd.values())

liab = stress.LiabilityBook(
    cashflows=np.concatenate([np.full(10, 380.0), np.full(20, 300.0),
                              np.full(10, 150.0)]))
curve_fn = lambda t: ns_loadings(t, model.lam) @ model.factors.iloc[-1].to_numpy() / 100

gap = stress.duration_gap(pf, liab, curve_fn, asset_dur)
print("Duration gap analysis (EUR mn):")
for k, v in gap.items():
    print(f"  {k:>22}: {v}")

print("\nKey-rate surplus sensitivity (per-tenor gap, asset-MV base):")
print(stress.krd_gap(pf, liab, curve_fn).to_string())

scenarios = {
    "+100bp parallel":      {"rates": {k: 100 for k in pf.krd}},
    "Bear steepener":       {"rates": {2: 20, 5: 50, 10: 90, 20: 110, 30: 120}},
    "Taper tantrum '13":    {"rates": {k: 80 for k in pf.krd},
                             "spreads_bp": 60, "equity_pct": -6, "fx_pct": -8},
    "Credit blowout (GFC)": {"spreads_bp": 250, "equity_pct": -35,
                             "rates": {k: -60 for k in pf.krd}},
    "Asia FX crisis":       {"fx_pct": -20, "spreads_bp": 120,
                             "equity_pct": -15,
                             "rates": {2: 150, 5: 120, 10: 90, 20: 70, 30: 60}},
}
res = stress.run_scenarios(pf, scenarios)
print("\nScenario P&L (EUR mn / % of MV):")
print(res.to_string())

cap = stress.capital_proxy(pf)
print("\nMarket-risk capital proxy (illustrative Solvency-style):")
for k, v in cap.items():
    print(f"  {k:>18}: {v}")

fig, ax = plt.subplots(figsize=(10, 4.5))
res_plot = res.drop(columns=["total_pct"])
res_plot[["rates", "spreads", "equity", "fx"]].plot(
    kind="bar", stacked=True, ax=ax,
    color=["#4477AA", "#EE6677", "#228833", "#CCBB44"])
ax.plot(range(len(res)), res["total"], "kD", label="total")
ax.axhline(0, color="k", lw=0.7)
ax.set_ylabel("P&L (EUR mn)"); ax.set_title("Stress scenario P&L decomposition")
ax.legend(); plt.xticks(rotation=15)
fig.tight_layout(); fig.savefig(f"{OUT}/3_stress.png", dpi=130)

print("\nCharts saved to outputs/. Done.")
