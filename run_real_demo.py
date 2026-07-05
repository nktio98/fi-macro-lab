"""End-to-end demo on REAL data (FRED, no API key): US Treasury curve ->
DNS fit + forecast; S&P 500 + IG OAS -> regime detection; ALM stress on
the live curve. Falls back to the simulators if offline."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aim_toolkit import data_live, stress
from aim_toolkit.regimes import JumpModel, regime_summary
from aim_toolkit.yield_curve import DNSModel, ns_loadings

OUT = "outputs"


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


section("1. DNS ON THE LIVE US TREASURY CURVE (FRED)")
yields = data_live.us_treasury_curve(start="2005-01-01")
print(f"Panel: {yields.index[0]:%Y-%m} .. {yields.index[-1]:%Y-%m}, "
      f"{yields.shape[0]} months x {yields.shape[1]} maturities")
model = DNSModel().fit(yields)
print(f"Optimal lambda: {model.lam:.3f} | in-sample RMSE: {model.rmse_bp:.1f} bp")
print("Latest factors:", model.factors.iloc[-1].round(3).to_dict())
fc = model.forecast_curve(h=12)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
model.factors.plot(ax=ax[0], title="DNS factors — US Treasuries (real data)")
mats = yields.columns.to_numpy(float)
ax[1].plot(mats, yields.iloc[-1], "o-", label=f"curve {yields.index[-1]:%Y-%m}")
ax[1].plot(mats, fc.iloc[2], "s--", label="forecast +3m")
ax[1].plot(mats, fc.iloc[-1], "^--", label="forecast +12m")
ax[1].set_xlabel("maturity (yrs)"); ax[1].set_ylabel("%"); ax[1].legend()
ax[1].set_title("VAR(1) curve forecast")
fig.tight_layout(); fig.savefig(f"{OUT}/real_1_yield_curve.png", dpi=130)

section("2. REGIMES ON REAL S&P 500 + IG SPREADS")
mkt = data_live.market_snapshot(start="2015-01-01")
feat = np.column_stack([
    mkt["equity_ret"].rolling(10).std().bfill(),
    mkt["equity_ret"].rolling(10).mean().bfill(),
    mkt["credit_spread_bp"].diff().rolling(10).mean().bfill(),
])
jm = JumpModel(jump_penalty=80.0).fit(feat)
print(regime_summary(jm.states, mkt["equity_ret"]).to_string())
print(f"Regime switches: {(np.diff(jm.states) != 0).sum()} "
      f"over {len(mkt)} days")

fig, ax = plt.subplots(figsize=(12, 4))
cum = (1 + mkt["equity_ret"]).cumprod()
ax.plot(mkt.index, cum, "k", lw=0.8)
ax.fill_between(mkt.index, cum.min(), cum.max(), where=jm.states == 1,
                alpha=0.25, color="red", label="stress regime")
ax.set_title("Jump-model regimes on real S&P 500"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/real_2_regimes.png", dpi=130)

section("3. ALM STRESS ON THE LIVE CURVE")
pf = stress.Portfolio(mv=10_000.0,
                      krd={2: 0.4, 5: 1.2, 10: 3.0, 20: 2.4, 30: 1.0},
                      spread_dur=5.5, credit_weight=0.45,
                      equity_weight=0.08, fx_unhedged_weight=0.05)
liab = stress.LiabilityBook(cashflows=np.concatenate(
    [np.full(10, 380.0), np.full(20, 300.0), np.full(10, 150.0)]))
curve_fn = lambda t: ns_loadings(t, model.lam) \
    @ model.factors.iloc[-1].to_numpy() / 100
gap = stress.duration_gap(pf, liab, curve_fn, asset_dur=sum(pf.krd.values()))
for k, v in gap.items():
    print(f"  {k:>22}: {v}")
print("\nKey-rate surplus sensitivity (live curve):")
print(stress.krd_gap(pf, liab, curve_fn).to_string())

print("\nDone. Charts in outputs/real_*.png")
