"""
AIM Strategist Dashboard -- Streamlit app over the aim_toolkit library.

Run locally :  streamlit run app.py
Deploy free :  push to GitHub -> share.streamlit.io -> pick repo/app.py

Live data comes from FRED (no key needed; set FRED_API_KEY in Streamlit
secrets for full-history spreads/equity). If FRED is unreachable the app
falls back to the offline simulators and says so.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from aim_toolkit import allocation as al
from aim_toolkit import data, data_live, fx, managers, stress, taa
from aim_toolkit.regimes import GaussianMS, JumpModel, regime_summary
from aim_toolkit.yield_curve import DNSModel, ns_loadings

st.set_page_config(page_title="AIM Strategist Dashboard", layout="wide")


# ------------------------------------------------------------- data layer
@st.cache_data(ttl=6 * 3600, show_spinner="Fetching US Treasury curve (FRED)...")
def get_yields():
    try:
        return data_live.us_treasury_curve(start="2005-01-01"), True
    except Exception:
        return data.simulate_yields(), False


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching market data (FRED)...")
def get_market():
    try:
        return data_live.market_snapshot(start="2015-01-01"), True
    except Exception:
        m = data.simulate_market()
        return m.assign(vix=np.nan), False


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching S&P 500 (FRED)...")
def get_equity():
    try:
        return data_live.equity_and_vix(start="2015-01-01"), True
    except Exception:
        m = data.simulate_market()
        px = (1 + m["equity_ret"]).cumprod() * 100
        return pd.DataFrame({"SP500": px, "VIX": 20.0}), False


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching FX (FRED)...")
def get_fx():
    try:
        return data_live.fx_rates(start="2000-01-01"), True
    except Exception:
        return None, False


@st.cache_data(show_spinner="Fitting DNS model...")
def fit_dns(yields: pd.DataFrame):
    return DNSModel().fit(yields)


def live_badge(is_live: bool, src: str):
    if is_live:
        st.caption(f":green[● LIVE] {src} via FRED, cached ≤6h")
    else:
        st.caption(f":orange[● OFFLINE] simulated {src} (FRED unreachable)")


# ------------------------------------------------------------------ header
st.title("AIM Strategist Dashboard")
st.caption("Yield curves · regimes · ALM stress · FX · TAA · manager "
           "oversight · allocation — insurance investment-strategist toolkit")

TAB_NAMES = ["Yield curve", "Regimes", "ALM stress", "FX", "TAA",
             "Managers", "Allocation"]
tabs = st.tabs(TAB_NAMES)

yields, yl_live = get_yields()
dns = fit_dns(yields)


# ------------------------------------------------------------- yield curve
with tabs[0]:
    live_badge(yl_live, "US Treasury constant-maturity yields")
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal λ", f"{dns.lam:.3f}")
    c2.metric("In-sample RMSE", f"{dns.rmse_bp:.1f} bp")
    c3.metric("10y yield (latest)", f"{yields.iloc[-1].get(10.0, np.nan):.2f} %")

    h = st.slider("Forecast horizon (months)", 1, 24, 12)
    fc = dns.forecast_curve(h)
    mats = yields.columns.to_numpy(float)

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(dns.factors, height=320)
        st.caption("DNS level / slope / curvature factors")
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(mats, yields.iloc[-1], "o-", label=f"current ({yields.index[-1]:%Y-%m})")
        ax.plot(mats, fc.iloc[min(2, h - 1)], "s--", label=f"+{min(3, h)}m forecast")
        ax.plot(mats, fc.iloc[-1], "^--", label=f"+{h}m forecast")
        ax.set_xlabel("maturity (yrs)"); ax.set_ylabel("%"); ax.legend()
        ax.set_title("VAR(1) curve forecast")
        st.pyplot(fig, clear_figure=True)
    st.caption("Long-run VAR factor means: "
               + np.array2string(dns.var.long_run_mean(), precision=2))


# ----------------------------------------------------------------- regimes
with tabs[1]:
    mkt, mkt_live = get_market()
    live_badge(mkt_live, "S&P 500 returns + US IG OAS")
    pen = st.slider("Jump penalty (higher = more persistent regimes)",
                    5.0, 300.0, 80.0, 5.0)
    ret = mkt["equity_ret"]
    feat = np.column_stack([
        ret.rolling(10).std().bfill(),
        ret.rolling(10).mean().bfill(),
        mkt["credit_spread_bp"].diff().rolling(10).mean().bfill(),
    ])
    jm = JumpModel(jump_penalty=pen).fit(feat)
    ms = GaussianMS().fit(ret.to_numpy())
    ms_states = (ms.smoothed[:, 1] > 0.5).astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    cum = (1 + ret).cumprod()
    for a, s_, name in [(axes[0], jm.states, f"Jump model (penalty {pen:.0f})"),
                        (axes[1], ms_states, "2-state Markov-switching")]:
        a.plot(mkt.index, cum, "k", lw=0.8)
        a.fill_between(mkt.index, cum.min(), cum.max(), where=s_ == 1,
                       alpha=0.25, color="red")
        a.set_title(f"{name} — shaded = stress regime")
    st.pyplot(fig, clear_figure=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Per-regime stats (jump model)**")
        st.dataframe(regime_summary(jm.states, ret))
    with col2:
        st.write("**Model comparison**")
        st.dataframe(pd.DataFrame({
            "switches": [(np.diff(jm.states) != 0).sum(),
                         (np.diff(ms_states) != 0).sum()],
            "stress freq %": [round(jm.states.mean() * 100, 1),
                              round(ms_states.mean() * 100, 1)],
        }, index=["Jump model", "MS-HMM"]))
        st.caption("MS-HMM expected regime duration (days): "
                   + np.array2string(ms.expected_duration, precision=0))


# -------------------------------------------------------------- ALM stress
with tabs[2]:
    live_badge(yl_live, "discounting on the fitted live curve")
    st.write("**Portfolio** (edit and everything below recomputes)")
    c = st.columns(5)
    mv = c[0].number_input("Assets MV (mn)", 1000.0, 1e6, 10_000.0, step=500.0)
    credit_w = c[1].number_input("Credit weight", 0.0, 1.0, 0.45, 0.05)
    spread_dur = c[2].number_input("Spread duration", 0.0, 15.0, 5.5, 0.5)
    eq_w = c[3].number_input("Equity weight", 0.0, 1.0, 0.08, 0.01)
    fx_w = c[4].number_input("Unhedged FX weight", 0.0, 1.0, 0.05, 0.01)

    krd_df = st.data_editor(pd.DataFrame(
        {"tenor": [2, 5, 10, 20, 30], "krd": [0.4, 1.2, 3.0, 2.4, 1.0]}),
        hide_index=True, width=350)
    pf = stress.Portfolio(mv=mv,
                          krd=dict(zip(krd_df["tenor"], krd_df["krd"])),
                          spread_dur=spread_dur, credit_weight=credit_w,
                          equity_weight=eq_w, fx_unhedged_weight=fx_w)

    lc = st.columns(3)
    cf1 = lc[0].number_input("Liab CF yrs 1-10 (mn/yr)", 0.0, 1e5, 380.0)
    cf2 = lc[1].number_input("Liab CF yrs 11-30 (mn/yr)", 0.0, 1e5, 300.0)
    cf3 = lc[2].number_input("Liab CF yrs 31-40 (mn/yr)", 0.0, 1e5, 150.0)
    liab = stress.LiabilityBook(cashflows=np.concatenate(
        [np.full(10, cf1), np.full(20, cf2), np.full(10, cf3)]))
    curve_fn = lambda t: ns_loadings(t, dns.lam) \
        @ dns.factors.iloc[-1].to_numpy() / 100

    gap = stress.duration_gap(pf, liab, curve_fn,
                              asset_dur=sum(pf.krd.values()))
    m = st.columns(4)
    m[0].metric("Liability PV (mn)", f"{gap['liab_pv']:,.0f}")
    m[1].metric("Duration gap (yrs)", f"{gap['dur_gap']:.2f}")
    m[2].metric("Economic surplus (mn)", f"{gap['surplus']:,.0f}")
    m[3].metric("Surplus Δ +100bp (mn)", f"{gap['surplus_chg_+100bp']:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Key-rate surplus sensitivity**")
        st.dataframe(stress.krd_gap(pf, liab, curve_fn))
    with col2:
        st.write("**Capital proxy** (illustrative, NOT regulatory)")
        st.dataframe(pd.Series(stress.capital_proxy(pf), name="value"))

    scenarios = {
        "+100bp parallel": {"rates": {k: 100 for k in pf.krd}},
        "Bear steepener": {"rates": {2: 20, 5: 50, 10: 90, 20: 110, 30: 120}},
        "Taper tantrum '13": {"rates": {k: 80 for k in pf.krd},
                              "spreads_bp": 60, "equity_pct": -6, "fx_pct": -8},
        "Credit blowout (GFC)": {"spreads_bp": 250, "equity_pct": -35,
                                 "rates": {k: -60 for k in pf.krd}},
        "Asia FX crisis": {"fx_pct": -20, "spreads_bp": 120, "equity_pct": -15,
                           "rates": {2: 150, 5: 120, 10: 90, 20: 70, 30: 60}},
    }
    res = stress.run_scenarios(pf, scenarios)
    fig, ax = plt.subplots(figsize=(10, 4))
    res[["rates", "spreads", "equity", "fx"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    ax.plot(range(len(res)), res["total"], "kD", label="total")
    ax.axhline(0, color="k", lw=0.7); ax.legend()
    ax.set_ylabel("P&L (mn)"); plt.xticks(rotation=12)
    ax.set_title("Scenario P&L decomposition")
    st.pyplot(fig, clear_figure=True)
    st.dataframe(res)


# ------------------------------------------------------------------- FX
with tabs[3]:
    fxr, fx_live = get_fx()
    live_badge(fx_live, "Asian FX vs USD")
    st.write("**Hedged yield pickup** — hedged USD credit vs local bonds "
             "(edit the assumptions)")
    mk = st.data_editor(pd.DataFrame({
        "market": ["SGD", "JPY", "KRW", "TWD"],
        "usd_asset_yield": [5.3, 5.3, 5.3, 5.3],
        "r_usd": [4.3, 4.3, 4.3, 4.3],
        "r_local": [3.6, 0.3, 3.0, 1.6],
        "local_asset_yield": [3.1, 1.0, 3.6, 1.7],
        "basis_bp": [-45, -60, -35, -80],
    }), hide_index=True)
    table = fx.pickup_table(
        {r["market"]: {k: r[k] for k in
                       ("usd_asset_yield", "r_usd", "r_local",
                        "local_asset_yield", "basis_bp")}
         for _, r in mk.iterrows()})
    st.dataframe(table)
    st.caption("pickup > 0: hedged USD credit beats local bonds for that "
               "investor. Negative x-ccy basis makes USD hedging costlier.")

    if fxr is not None:
        eq, _ = get_equity()
        ccy = st.selectbox("Investor currency", list(fxr.columns), index=0)
        spot = fxr[ccy].dropna()

        st.write(f"**Fair value (Engle-Granger ECM)** — log USD/{ccy} vs "
                 "US 10y yield (BEER-lite, illustrative)")
        us10 = yields[10.0].dropna() if 10.0 in yields.columns else None
        if us10 is not None:
            m_spot = np.log(spot.resample("ME").last()).dropna()
            fund = pd.DataFrame({"us10y": us10}).reindex(m_spot.index).dropna()
            m_spot = m_spot.loc[fund.index]
            ecm = fx.ECMFairValue().fit(m_spot, fund)
            fv = ecm.fair_value(fund)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(m_spot.index, m_spot, label=f"log USD/{ccy}")
            ax.plot(fv.index, fv, "--", label="ECM fair value")
            sd = ecm.resid.std()
            ax.fill_between(fv.index, fv - 2 * sd, fv + 2 * sd, alpha=0.15)
            ax.legend(); ax.set_title(ecm.summary().replace("\n", " | "))
            st.pyplot(fig, clear_figure=True)

        st.write("**Rolling minimum-variance hedge ratio** — S&P 500 held "
                 f"by a {ccy} investor")
        fx_ret = spot.pct_change()                     # local per USD
        eq_usd = eq["SP500"].pct_change().reindex(spot.index).dropna()
        idx = eq_usd.index.intersection(fx_ret.index)
        unhedged = (1 + eq_usd.loc[idx]) * (1 + fx_ret.loc[idx]) - 1
        h = fx.min_var_hedge_ratio(unhedged, fx_ret.loc[idx], window=126)
        st.line_chart(h.dropna().clip(-0.5, 2), height=260)
        st.caption("h* = Cov(unhedged return, FX return)/Var(FX). "
                   "1 = full hedge of the USD exposure.")


# ------------------------------------------------------------------- TAA
with tabs[4]:
    eq, eq_live = get_equity()
    live_badge(eq_live, "S&P 500")
    prices = eq["SP500"].dropna()
    asset_ret = prices.pct_change().dropna()
    prices = prices.loc[asset_ret.index]

    def make_position(inputs, lookback, skip):
        sig = taa.momentum(inputs["prices"], lookback=lookback, skip=skip)
        return taa.zscore_position(sig).fillna(0.0)

    grid = [{"lookback": lb, "skip": sk}
            for lb in (63, 126, 252) for sk in (5, 21)]
    cv = taa.PurgedKFold(n_splits=5, label_horizon=21, embargo_pct=0.02)
    with st.spinner("Running purged CV over the signal grid..."):
        res, sel = taa.cv_sharpes(make_position, grid, {"prices": prices},
                                  asset_ret, cv)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Signal grid — purged CV** (train-selected, test-scored)")
        st.dataframe(res.sort_values("test_sharpe_mean", ascending=False))
    with col2:
        st.write("**Selection procedure, out of sample**")
        st.dataframe(sel)
        st.metric("OOS Sharpe of selection", f"{sel['oos_sharpe'].mean():.2f}")

    best = res.sort_values("test_sharpe_mean", ascending=False).iloc[0]
    pos = make_position({"prices": prices},
                        int(best["lookback"]), int(best["skip"]))
    net = taa.backtest(pos, asset_ret)["net"]
    grid_srs = np.array([taa.sharpe(taa.backtest(
        make_position({"prices": prices}, g["lookback"], g["skip"]),
        asset_ret)["net"]) for g in grid]) / np.sqrt(taa.ANN)
    dsr = taa.deflated_sharpe(net, n_trials=len(grid),
                              trial_sr_var=float(grid_srs.var()))
    m = st.columns(3)
    m[0].metric("Best config net Sharpe", f"{taa.sharpe(net):.2f}")
    m[1].metric("Probabilistic Sharpe", f"{taa.probabilistic_sharpe(net):.0%}")
    m[2].metric(f"Deflated Sharpe ({len(grid)} trials)", f"{dsr:.0%}")
    verdict = ("✅ survives multiple-testing correction" if dsr > 0.95
               else "⚠️ NOT proven — likely selection bias; do not deploy "
                    "on this evidence")
    st.info(f"Verdict: {verdict}")
    st.line_chart((1 + net).cumprod(), height=260)


# --------------------------------------------------------------- managers
with tabs[5]:
    st.caption(":blue[● SYNTHETIC] manager returns are simulated "
               "(no public manager data); factors and machinery are real")
    rng = np.random.default_rng(11)
    T, n_mgr = 120, 20
    fac = pd.DataFrame(rng.normal(0.004, 0.03, (T, 2)),
                       columns=["mkt", "credit"],
                       index=pd.date_range("2016-01-31", periods=T, freq="ME"))
    true_alpha = np.zeros(n_mgr); true_alpha[[3, 11]] = 0.0025   # 2 skilled
    rows, rets = [], {}
    for i in range(n_mgr):
        b = rng.uniform(0.6, 1.1), rng.uniform(0.0, 0.5)
        r = true_alpha[i] + b[0] * fac["mkt"] + b[1] * fac["credit"] \
            + rng.normal(0, 0.008, T)
        rets[f"mgr_{i:02d}"] = r
        out = managers.factor_regression(r, fac)
        out["pval"] = managers.alpha_pvalue_from_t(out["alpha_t"], T - 3)
        rows.append({"manager": f"mgr_{i:02d}",
                     "alpha_ann_%": round(out["alpha_ann_%"], 2),
                     "alpha_t": round(out["alpha_t"], 2),
                     "pval": round(out["pval"], 4), "r2": round(out["r2"], 2)})
    panel = pd.DataFrame(rows).set_index("manager")
    bh = managers.benjamini_hochberg(panel["pval"], fdr=0.10)
    panel["BH significant (FDR 10%)"] = bh["significant_at_FDR"]
    naive = (panel["alpha_t"].abs() > 2).sum()
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write("**Alpha panel with Newey-West t-stats + BH-FDR control**")
        st.dataframe(panel.sort_values("alpha_t", ascending=False), height=420)
    with col2:
        st.metric("Naive |t|>2 'skilled'", int(naive))
        st.metric("BH-FDR survivors", int(panel["BH significant (FDR 10%)"].sum()))
        st.caption("True skilled managers in the simulation: 2 "
                   "(mgr_03, mgr_11). Naive screens over-hire; FDR control "
                   "is the fix.")
        pick = st.selectbox("Style-drift monitor", list(rets))
        rb = managers.rolling_betas(pd.Series(rets[pick], index=fac.index),
                                    fac, window=36)
        st.line_chart(rb, height=220)


# -------------------------------------------------------------- allocation
with tabs[6]:
    mkt, mkt_live = get_market()
    live_badge(mkt_live, "scenario engine inputs (equity, rates, credit)")
    y10 = yields[10.0].resample("ME").last() if 10.0 in yields.columns else None
    eq_m = (1 + mkt["equity_ret"]).resample("ME").prod() - 1
    oas = mkt["credit_spread_bp"].resample("ME").last()
    dy = y10.diff().reindex(eq_m.index)
    doas = oas.diff()
    govt_m = (y10.reindex(eq_m.index) / 12 - 8.0 * dy) / 100
    cred_m = govt_m + (oas.reindex(eq_m.index) / 12 - 5.5 * doas) / 1e4
    hist = pd.DataFrame({"govt": govt_m, "credit": cred_m,
                         "equity": eq_m}).dropna()
    st.caption(f"Monthly joint scenarios from {hist.index[0]:%Y-%m} to "
               f"{hist.index[-1]:%Y-%m} ({len(hist)} obs), bootstrapped to "
               "1,000 scenarios. Sleeves: govt (10y, dur 8), IG credit "
               "(dur 5.5), equity.")
    rng = np.random.default_rng(5)
    scen = hist.to_numpy()[rng.integers(0, len(hist), 1000)]

    view_eq = st.slider("View: expected EQUITY return (% p.a.)",
                        -10.0, 15.0, 4.0, 0.5) / 100 / 12
    view_cr = st.slider("View: expected CREDIT return (% p.a.)",
                        -5.0, 10.0, 4.5, 0.5) / 100 / 12
    ep = al.EntropyPooling().fit(
        scen,
        np.vstack([al.view_on_mean(scen, 2), al.view_on_mean(scen, 1)]),
        [view_eq, view_cr])
    mu_post, Sigma_post = ep.posterior_moments()

    ra = st.slider("Risk aversion", 1.0, 10.0, 4.0, 0.5)
    w = al.mv_optimize(mu_post, Sigma_post, risk_aversion=ra, w_max=0.7)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Posterior (entropy-pooled) moments, monthly**")
        st.dataframe(pd.DataFrame(
            {"E[r] % p.a.": np.round(mu_post * 12 * 100, 2),
             "vol % p.a.": np.round(np.sqrt(np.diag(Sigma_post) * 12) * 100, 2),
             "weight": np.round(w, 3)}, index=hist.columns))
        st.metric("Effective scenarios (ENS)", f"{ep.effective_n:,.0f} / 1,000")
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(hist.columns, w, color=["#4477AA", "#EE6677", "#228833"])
        ax.set_title("View-conditioned allocation (long-only MV)")
        ax.set_ylabel("weight")
        st.pyplot(fig, clear_figure=True)
        st.caption("Pipeline: historical scenarios → entropy-pooling tilt "
                   "to the view → constrained optimizer. Low ENS = the view "
                   "is fighting the data.")
