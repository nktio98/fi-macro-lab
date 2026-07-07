"""
AIM Strategist Dashboard -- Asia-focused, multi-economy.

Designed around three responsibilities of a regional investment strategist:
  1. Economic & capital-market analysis (rates, FX, macro drivers,
     regime shifts)          -> tabs "Rates & curves", "FX & regimes"
  2. Translate macro views into actionable TAA proposals
                             -> tab  "Strategy & TAA"
  3. Forward-looking scenario analysis & stress testing (rates, spreads,
     FX, geopolitical)       -> tab  "Stress & resilience"
plus asset-manager oversight -> tab  "Manager oversight".

Data: AsianBondsOnline (ASEAN+3 LCY curves incl. Indonesia & Singapore),
Japan MoF (full JGB history), ECB (euro AAA curve, IDR/INR/TWD FX),
FRED (US curve, OECD 10y panel, Asian FX, spreads, S&P 500). All free,
no API key; cached 24h on disk + 6h in-app. Offline fallback: simulators.

Run locally :  streamlit run app.py
Deploy free :  push to GitHub -> share.streamlit.io -> pick repo/app.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from aim_toolkit import allocation as al
from aim_toolkit import data, data_global, data_live, fx, macro, managers, \
    monitoring, nowcast, stress, taa
from aim_toolkit.regimes import GaussianMS, JumpModel, regime_summary
from aim_toolkit.yield_curve import ACMTermPremium, DNSModel, ns_loadings

st.set_page_config(page_title="AIM Strategist Dashboard — Asia",
                   layout="wide")

ASIA_NAMES = data_global.ABO_ECONOMIES


# ------------------------------------------------------------- data layer
@st.cache_data(ttl=6 * 3600, show_spinner="Fetching US Treasury curve...")
def get_us_curve():
    try:
        return data_live.us_treasury_curve(start="2005-01-01"), True
    except Exception:
        return data.simulate_yields(), False


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching JGB curve (MoF)...")
def get_jgb_curve():
    try:
        return data_global.jgb_curve(start="2005-01-01"), True
    except Exception:
        return None, False


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching euro AAA curve (ECB)...")
def get_ecb_curve():
    try:
        return data_global.ecb_curve(), True
    except Exception:
        return None, False


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching Asian curves (ADB)...")
def get_asia_curves():
    try:
        out = data_global.abo_curves()
        return ({k: v for k, v in out.items()},
                {k: v.attrs.get("as_of", "") for k, v in out.items()})
    except Exception:
        return {}, {}


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching global 10y panel...")
def get_10y_panel():
    try:
        return data_global.global_10y_panel(start="2005-01-01")
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching FX (FRED + ECB)...")
def get_fx_all():
    frames = []
    try:
        frames.append(data_live.fx_rates(start="2005-01-01"))
    except Exception:
        pass
    try:
        frames.append(data_global.ecb_fx_usd(("IDR", "INR", "TWD")))
    except Exception:
        pass
    if not frames:
        return None
    return pd.concat(frames, axis=1).sort_index().ffill()


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching market data (FRED)...")
def get_market():
    try:
        return data_live.market_snapshot(start="2015-01-01"), True
    except Exception:
        m = data.simulate_market()
        return m.assign(vix=np.nan), False


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching S&P 500...")
def get_equity():
    try:
        return data_live.equity_and_vix(start="2015-01-01"), True
    except Exception:
        m = data.simulate_market()
        px = (1 + m["equity_ret"]).cumprod() * 100
        return pd.DataFrame({"SP500": px, "VIX": 20.0}), False


@st.cache_data(show_spinner="Fitting DNS model...")
def fit_dns(yields: pd.DataFrame):
    return DNSModel().fit(yields)


def live_badge(is_live: bool, src: str):
    if is_live:
        st.caption(f":green[● LIVE] {src}")
    else:
        st.caption(f":orange[● OFFLINE] simulated stand-in for {src}")


def snap_yield(snap: pd.DataFrame, tenor: float) -> float:
    """Interpolated yield at a tenor from an ABO snapshot."""
    return float(np.interp(tenor, snap.index.to_numpy(float),
                           snap["yield_pct"].to_numpy(float)))


# ------------------------------------------------------------------ header
st.title("AIM Strategist Dashboard — Asia")
st.caption("Multi-economy rates · FX · regimes · TAA · ALM stress · manager "
           "oversight — built for regional insurance portfolio strategy")

tabs = st.tabs(["🌏 Rates & curves", "💱 FX & regimes", "📊 Macro lab",
                "🎯 Strategy & TAA", "🛡️ Stress & resilience",
                "🔍 Manager oversight"])

us_yields, us_live = get_us_curve()
dns_us = fit_dns(us_yields)


# =========================================================== rates & curves
with tabs[0]:
    st.caption("*Objective: economic & capital-market analysis — interest "
               "rates: trends, curve shapes, and cross-market comparison.*")

    asia, as_ofs = get_asia_curves()
    st.subheader("Asian LCY government curves (AsianBondsOnline, ADB)")
    if asia:
        default = [c for c in ("ID", "SG", "KR", "CN", "TH", "MY", "JP", "US")
                   if c in asia]
        sel = st.multiselect("Economies", list(asia),
                             default=default,
                             format_func=lambda c: ASIA_NAMES.get(c, c))
        col1, col2 = st.columns([3, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for c in sel:
                s = asia[c]
                ax.plot(s.index, s["yield_pct"], "o-", ms=3.5,
                        label=ASIA_NAMES.get(c, c))
            ax.set_xlabel("maturity (yrs)"); ax.set_ylabel("yield (%)")
            any_asof = next((v for v in as_ofs.values() if v), "")
            ax.set_title(f"LCY government yield curves — close {any_asof}")
            ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        with col2:
            rows = {}
            for c in sel:
                s = asia[c]
                y10, y2 = snap_yield(s, 10), snap_yield(s, 2)
                ytd = s["ytd_bp"].iloc[
                    int(np.abs(s.index.to_numpy(float) - 10).argmin())]
                rows[ASIA_NAMES.get(c, c)] = {
                    "10y %": round(y10, 2), "2s10s bp": round((y10 - y2) * 100, 0),
                    "10y YTD bp": round(ytd, 0)}
            st.dataframe(pd.DataFrame(rows).T
                         .sort_values("10y %", ascending=False))
            st.caption("2s10s from interpolated snapshot tenors. "
                       "Steep + high-yield (ID) vs flat + low-yield (JP, CN) "
                       "is the regional carry map at a glance.")
    else:
        st.warning("AsianBondsOnline unreachable — Asian curve snapshot "
                   "unavailable this session.")

    st.divider()
    st.subheader("Curve dynamics lab — DNS factors & VAR forecast")
    jgb, jgb_live = get_jgb_curve()
    ecb, ecb_live = get_ecb_curve()
    panels = {"United States (UST)": (us_yields if us_live else None),
              "Japan (JGB)": jgb, "Euro area (AAA)": ecb}
    avail = {k: v for k, v in panels.items() if v is not None}
    if not avail:
        avail = {"United States (simulated)": us_yields}
    pick = st.selectbox("Market (deep-history panels)", list(avail))
    ylds = avail[pick]
    model = fit_dns(ylds)
    h = st.slider("Forecast horizon (months)", 1, 24, 12)
    fc = model.forecast_curve(h)
    mats = ylds.columns.to_numpy(float)
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal λ", f"{model.lam:.3f}")
    c2.metric("In-sample RMSE", f"{model.rmse_bp:.1f} bp")
    c3.metric("10y (latest)", f"{np.interp(10, mats, ylds.iloc[-1]):.2f} %")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(model.factors, height=300)
        st.caption("Level / slope / curvature — level ≈ long-run rate view, "
                   "slope ≈ policy stance, curvature ≈ belly positioning")
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(mats, ylds.iloc[-1], "o-",
                label=f"current ({ylds.index[-1]:%Y-%m})")
        ax.plot(mats, fc.iloc[min(2, h - 1)], "s--", label=f"+{min(3, h)}m")
        ax.plot(mats, fc.iloc[-1], "^--", label=f"+{h}m")
        ax.set_xlabel("maturity (yrs)"); ax.set_ylabel("%")
        ax.set_title("VAR(1) factor-forecast curve"); ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("Global 10y government yields — history")
    g10 = get_10y_panel()
    if g10 is not None:
        cols = st.multiselect("Markets", list(g10.columns),
                              default=list(g10.columns))
        st.line_chart(g10[cols], height=320)
        st.caption("Monthly OECD long-term yields via FRED. Divergence "
                   "US/EU vs JP/KR is the hedged-yield-pickup driver on "
                   "the Strategy tab.")

    st.divider()
    st.subheader("US term premium — ACM decomposition")
    if us_live:
        @st.cache_data(show_spinner="Fitting ACM term-premium model...")
        def fit_acm(y: pd.DataFrame):
            return ACMTermPremium().fit(y, k_factors=3, max_years=10)
        acm = fit_acm(us_yields)
        fitted10 = acm.fitted_yield(10)
        rn10 = acm.risk_neutral_yield(10)
        tp10 = acm.term_premium(10)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("10y model yield", f"{fitted10.iloc[-1]:.2f} %")
        c2.metric("Expectations part", f"{rn10.iloc[-1]:.2f} %")
        c3.metric("Term premium", f"{tp10.iloc[-1] * 100:+.0f} bp")
        c4.metric("Pricing RMSE (10y)", f"{acm.fit_rmse_bp(10):.0f} bp")
        st.line_chart(pd.DataFrame({
            "10y yield (model)": fitted10,
            "expected avg short rate": rn10,
            "term premium": tp10}), height=320)
        st.caption("Adrian-Crump-Moench 3-step estimator on the live UST "
                   "panel. High/positive term premium = you are PAID to "
                   "extend duration beyond the pure rate view; negative "
                   "premium = duration is expensive insurance. The "
                   "expectations line is the market-implied average short "
                   "rate over 10y — compare against your own Fed view.")
    else:
        st.info("ACM needs the live US panel.")


# ============================================================ fx & regimes
with tabs[1]:
    st.caption("*Objective: economic & capital-market analysis — FX and "
               "macro drivers: trends, regime shifts, valuation signals.*")

    fxr = get_fx_all()
    eq, eq_live = get_equity()
    st.subheader("Asian FX monitor (vs USD)")
    if fxr is not None:
        ccys = st.multiselect("Currencies", list(fxr.columns),
                              default=[c for c in ("SGD", "JPY", "KRW", "IDR",
                                                   "THB", "MYR", "CNY")
                                       if c in fxr.columns])
        lookback = st.radio("Window", ["1y", "3y", "10y"], index=1,
                            horizontal=True)
        n = {"1y": 252, "3y": 756, "10y": 2520}[lookback]
        sub = fxr[ccys].dropna(how="all").iloc[-n:]
        norm = sub / sub.iloc[0] * 100
        col1, col2 = st.columns([3, 2])
        with col1:
            st.line_chart(norm, height=320)
            st.caption("Indexed to 100 at window start; UP = local currency "
                       "DEPRECIATING vs USD (rates are local per USD).")
        with col2:
            last = fxr[ccys].dropna(how="all")
            ytd_start = last.loc[last.index >= f"{last.index[-1].year}-01-01"]
            tbl = pd.DataFrame({
                "spot": last.iloc[-1].round(2),
                "YTD %": ((last.iloc[-1] / ytd_start.iloc[0] - 1) * 100).round(2),
            })
            st.dataframe(tbl)
            st.caption("Positive YTD % = depreciation vs USD, i.e. an "
                       "unhedged USD asset gained in local terms.")

    st.divider()
    st.subheader("Regime detection")
    mkt, mkt_live = get_market()
    assets = {"S&P 500 (global risk)": None}
    if fxr is not None:
        for c in fxr.columns:
            assets[f"USD/{c}"] = c
    pick = st.selectbox("Series", list(assets))
    pen = st.slider("Jump penalty (persistence)", 5.0, 300.0, 80.0, 5.0)
    if assets[pick] is None:
        ret = mkt["equity_ret"]
        feat = np.column_stack([
            ret.rolling(10).std().bfill(),
            ret.rolling(10).mean().bfill(),
            mkt["credit_spread_bp"].diff().rolling(10).mean().bfill()])
    else:
        ret = fxr[assets[pick]].pct_change().dropna().iloc[-2520:]
        feat = np.column_stack([ret.rolling(10).std().bfill(),
                                ret.rolling(10).mean().bfill()])
    jm = JumpModel(jump_penalty=pen).fit(feat)
    ms = GaussianMS().fit(ret.to_numpy())
    ms_states = (ms.smoothed[:, 1] > 0.5).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    cum = (1 + ret).cumprod()
    for a, s_, name in [(axes[0], jm.states, f"Jump model (penalty {pen:.0f})"),
                        (axes[1], ms_states, "2-state Markov-switching")]:
        a.plot(ret.index, cum, "k", lw=0.8)
        a.fill_between(ret.index, cum.min(), cum.max(), where=s_ == 1,
                       alpha=0.25, color="red")
        a.set_title(f"{name} — shaded = stress/volatile regime")
    st.pyplot(fig, clear_figure=True)
    col1, col2 = st.columns(2)
    col1.dataframe(regime_summary(jm.states, ret))
    col2.dataframe(pd.DataFrame({
        "switches": [(np.diff(jm.states) != 0).sum(),
                     (np.diff(ms_states) != 0).sum()],
        "stress freq %": [round(jm.states.mean() * 100, 1),
                          round(ms_states.mean() * 100, 1)]},
        index=["Jump model", "MS-HMM"]))

    st.divider()
    st.subheader("FX fair value — Engle-Granger ECM (BEER-lite)")
    if fxr is not None:
        ccy = st.selectbox("Currency", [c for c in fxr.columns], index=0,
                           key="ecm_ccy")
        spot = fxr[ccy].dropna()
        m_spot = np.log(spot.resample("ME").last()).dropna()
        us10 = us_yields[10.0].resample("ME").last() \
            if 10.0 in us_yields.columns else None
        jgb_c, _ = get_jgb_curve()
        fund = pd.DataFrame(index=m_spot.index)
        if us10 is not None:
            if ccy == "JPY" and jgb_c is not None and 10.0 in jgb_c.columns:
                diff = (us10 - jgb_c[10.0].resample("ME").last())
                fund["rate_diff_10y"] = diff.reindex(m_spot.index)
            else:
                fund["us10y"] = us10.reindex(m_spot.index)
        fund = fund.dropna()
        if len(fund) > 48:
            m_spot2 = m_spot.loc[fund.index]
            ecm = fx.ECMFairValue().fit(m_spot2, fund)
            fv = ecm.fair_value(fund)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(m_spot2.index, m_spot2, label=f"log USD/{ccy}")
            ax.plot(fv.index, fv, "--", label="ECM fair value")
            sd = ecm.resid.std()
            ax.fill_between(fv.index, fv - 2 * sd, fv + 2 * sd, alpha=0.15)
            ax.legend(); ax.grid(alpha=0.3)
            ax.set_title(ecm.summary().replace("\n", "  |  "), fontsize=9)
            st.pyplot(fig, clear_figure=True)
            st.caption("Fundamental: US-Japan 10y differential for JPY, US "
                       "10y otherwise — illustrative BEER; a production "
                       "model adds ToT, CA balance, relative CPI. Outside "
                       "the ±2σ band = valuation signal; half-life = how "
                       "fast misvaluation historically decays.")

    st.divider()
    st.subheader("Minimum-variance hedge ratio")
    if fxr is not None:
        eqd, _ = get_equity()
        hc1, hc2 = st.columns(2)
        ccy_h = hc1.selectbox("Investor currency", list(fxr.columns),
                              index=0, key="hr_ccy")
        method = hc2.radio("Estimator", ["Rolling OLS (126d)", "DCC-GARCH"],
                           horizontal=True)
        spot_h = fxr[ccy_h].dropna().iloc[-2000:]
        fx_ret_h = spot_h.pct_change()
        eq_usd = eqd["SP500"].pct_change().reindex(spot_h.index)
        idxh = eq_usd.dropna().index.intersection(fx_ret_h.dropna().index)
        unhedged = ((1 + eq_usd.loc[idxh]) * (1 + fx_ret_h.loc[idxh]) - 1)
        if method.startswith("Rolling"):
            h = fx.min_var_hedge_ratio(unhedged, fx_ret_h.loc[idxh],
                                       window=126).dropna()
            st.line_chart(h.clip(-0.5, 2.5), height=260)
        else:
            with st.spinner("Fitting GARCH(1,1) ×2 + DCC by MLE..."):
                out = fx.dcc_hedge_ratio(unhedged, fx_ret_h.loc[idxh])
            st.line_chart(out["hedge_ratio"].clip(-0.5, 2.5), height=260)
            st.caption(f"DCC(1,1): a={out['a']:.3f}, b={out['b']:.3f} "
                       f"(persistence {out['a'] + out['b']:.3f}). "
                       "Conditional correlation × vol ratio — reacts to "
                       "regime change in days rather than the ~6 months a "
                       "rolling window needs.")
        st.caption(f"S&P 500 held by a {ccy_h} investor; h*=1 is a full "
                   "hedge of the USD exposure. When the estimate swings "
                   "materially, a static hedge policy leaves risk (or "
                   "return) on the table.")


# ============================================================== macro lab
with tabs[2]:
    st.caption("*Objective: economic analysis — where is the economy right "
               "now, and how do macro shocks transmit into markets?*")

    st.subheader("GDP nowcast — monthly activity factor + bridge")
    NC_NAMES = {"ID": "Indonesia", "KR": "Korea", "JP": "Japan",
                "US": "United States"}
    ec = st.selectbox("Economy", list(NC_NAMES),
                      format_func=lambda c: NC_NAMES[c])

    @st.cache_data(ttl=6 * 3600, show_spinner="Building nowcast...")
    def get_nowcast(economy: str):
        try:
            return nowcast.bridge_nowcast(economy)
        except Exception:
            return None

    nc = get_nowcast(ec)
    if nc is None:
        st.warning("Macro series unavailable (FRED unreachable or series "
                   "discontinued).")
    else:
        m = st.columns(4)
        m[0].metric("Activity factor",
                    f"{nc['latest_factor']:+.2f} σ",
                    help=f"as of {nc['factor_date']:%Y-%m}")
        m[1].metric("Bridge R²", f"{nc['r2']:.2f}")
        m[2].metric("Factor t-stat", f"{nc['t_stats'][1]:.1f}")
        if nc["nowcast"]:
            q, v = next(iter(nc["nowcast"].items()))
            m[3].metric(f"Nowcast {q.year}Q{q.quarter}", f"{v:+.2f}% q/q")
        else:
            m[3].metric("Nowcast", "GDP printed",
                        help="no pending quarter to nowcast")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.line_chart(nc["factor"], height=260)
            st.caption("Monthly activity factor (EM-PCA over the indicator "
                       "panel; +1σ = clearly above-trend momentum)")
        with col2:
            zlast = ((nc["panel"] - nc["panel"].mean())
                     / nc["panel"].std()).iloc[-1].round(2)
            st.dataframe(pd.DataFrame({"loading": nc["loadings"].round(2),
                                       "latest z": zlast}))
            st.caption("Indicator diagnostics. Low bridge R² (Indonesia) "
                       "is honest: free monthly data for EM Asia is thin — "
                       "CPI + exports + commodities, no PMI (licensed). "
                       "SG/MY/TH have no usable free series at all.")

    st.divider()
    st.subheader("Shock transmission — local projections (Jordà)")
    st.caption("If the **US 10y** moves +100bp this month, what happens "
               "over the following 12 months? One OLS per horizon, "
               "Newey-West bands — no structural VAR identification "
               "assumptions.")
    fxr_m, _ = get_market()
    fxx = get_fx_all()
    eqx, _ = get_equity()
    shock = us_yields[10.0].diff().dropna() if 10.0 in us_yields.columns \
        else None
    responses = {}
    if fxx is not None:
        for c in ("SGD", "IDR", "KRW", "JPY"):
            if c in fxx.columns:
                responses[f"USD/{c} (%)"] = \
                    np.log(fxx[c].resample("ME").last()).diff() * 100
    responses["US IG OAS (bp)"] = \
        fxr_m["credit_spread_bp"].resample("ME").last().diff()
    responses["S&P 500 (%)"] = \
        np.log(eqx["SP500"].resample("ME").last()).diff() * 100
    pick_r = st.selectbox("Response variable", list(responses))
    if shock is not None:
        irf = macro.local_projections(responses[pick_r], shock,
                                      horizons=12, lags=3)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(irf.index, irf["beta"], "o-", color="#4477AA")
        ax.fill_between(irf.index, irf["lo"], irf["hi"], alpha=0.2,
                        color="#4477AA")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("months after shock")
        ax.set_ylabel(f"cumulative response, {pick_r}")
        ax.set_title(f"{pick_r} response to +100bp US 10y shock")
        ax.grid(alpha=0.3)
        st.pyplot(fig, clear_figure=True)
        s = macro.irf_summary(irf)
        sig = "significant at 90%" if s["significant"] else \
            "NOT significant at 90% — treat as noise"
        st.caption(f"Peak response {s['peak_beta']:+.2f} at month "
                   f"{s['peak_h']} ({sig}). Monthly panel is short where "
                   "spreads are involved (free BAML data ≈ 3y without a "
                   "FRED key) — bands widen accordingly.")


# ========================================================== strategy & TAA
with tabs[3]:
    st.caption("*Objective: translate macro views, market developments and "
               "valuation signals into actionable TAA — consistent with "
               "objectives and constraints.*")

    st.subheader("Cross-market hedged yield pickup (live curves)")
    asia, _ = get_asia_curves()
    mkt_, _ = get_market()
    if asia and 1.0 in us_yields.columns:
        r_usd = float(us_yields[1.0].iloc[-1])
        us10 = float(np.interp(10, us_yields.columns.to_numpy(float),
                               us_yields.iloc[-1]))
        try:
            oas_bp = float(mkt_["credit_spread_bp"].iloc[-1])
        except Exception:
            oas_bp = 90.0
        usd_asset = us10 + oas_bp / 100
        basis_defaults = {"SG": -45, "ID": -80, "KR": -35, "TH": -40,
                          "MY": -30, "CN": -30, "JP": -60, "HK": -20,
                          "PH": -60, "VN": -80}
        rows = []
        for c in [e for e in ("SG", "ID", "KR", "TH", "MY", "CN", "JP")
                  if e in asia]:
            s = asia[c]
            rows.append({"market": ASIA_NAMES.get(c, c),
                         "usd_asset_yield": round(usd_asset, 2),
                         "r_usd": round(r_usd, 2),
                         "r_local": round(snap_yield(s, 1), 2),
                         "local_asset_yield": round(snap_yield(s, 10), 2),
                         "basis_bp": basis_defaults.get(c, -40)})
        st.caption(f"Live inputs: USD 1y {r_usd:.2f}%, USD IG asset yield "
                   f"{usd_asset:.2f}% (US 10y {us10:.2f}% + IG OAS "
                   f"{oas_bp:.0f}bp). Local 1y/10y from ABO curves. "
                   "X-ccy basis is editable (no free source).")
        mk = st.data_editor(pd.DataFrame(rows), hide_index=True)
        table = fx.pickup_table(
            {r["market"]: {k: r[k] for k in
                           ("usd_asset_yield", "r_usd", "r_local",
                            "local_asset_yield", "basis_bp")}
             for _, r in mk.iterrows()})
        st.dataframe(table)
        st.caption("pickup > 0: hedged USD IG credit beats the local 10y "
                   "govt bond for that investor — the core Asian insurance "
                   "allocation decision. High-carry markets (ID) usually "
                   "favour local bonds; low-yield markets (JP) depend "
                   "heavily on the basis.")
    else:
        st.warning("Needs live US curve + ABO curves; using the FX tab "
                   "editable table instead.")

    st.divider()
    st.subheader("TAA signal lab — momentum with honest validation")
    eq, eq_live = get_equity()
    live_badge(eq_live, "S&P 500 (global risk proxy)")
    prices = eq["SP500"].dropna()
    asset_ret = prices.pct_change().dropna()
    prices = prices.loc[asset_ret.index]

    def make_position(inputs, lookback, skip):
        sig = taa.momentum(inputs["prices"], lookback=lookback, skip=skip)
        return taa.zscore_position(sig).fillna(0.0)

    grid = [{"lookback": lb, "skip": sk}
            for lb in (63, 126, 252) for sk in (5, 21)]
    cv = taa.PurgedKFold(n_splits=5, label_horizon=21, embargo_pct=0.02)
    with st.spinner("Running purged CV..."):
        res, sel = taa.cv_sharpes(make_position, grid, {"prices": prices},
                                  asset_ret, cv)
    col1, col2 = st.columns(2)
    col1.dataframe(res.sort_values("test_sharpe_mean", ascending=False))
    with col2:
        st.dataframe(sel)
        st.metric("OOS Sharpe of selection", f"{sel['oos_sharpe'].mean():.2f}")
    best = res.sort_values("test_sharpe_mean", ascending=False).iloc[0]
    pos = make_position({"prices": prices}, int(best["lookback"]),
                        int(best["skip"]))
    net = taa.backtest(pos, asset_ret)["net"]
    grid_srs = np.array([taa.sharpe(taa.backtest(
        make_position({"prices": prices}, g["lookback"], g["skip"]),
        asset_ret)["net"]) for g in grid]) / np.sqrt(taa.ANN)
    dsr = taa.deflated_sharpe(net, n_trials=len(grid),
                              trial_sr_var=float(grid_srs.var()))
    m = st.columns(3)
    m[0].metric("Best net Sharpe", f"{taa.sharpe(net):.2f}")
    m[1].metric("Probabilistic Sharpe", f"{taa.probabilistic_sharpe(net):.0%}")
    m[2].metric(f"Deflated Sharpe ({len(grid)} trials)", f"{dsr:.0%}")
    st.info("Verdict: " + ("✅ survives multiple-testing correction"
                           if dsr > 0.95 else
                           "⚠️ NOT proven — likely selection bias; do not "
                           "deploy on this evidence"))

    st.divider()
    st.subheader("Signal governance — post-deployment decay monitor")
    rep = monitoring.decay_report(net.dropna(), window=252)
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Full-sample IR", f"{rep['full_ir']:.2f}")
    g2.metric("Recent IR (1y)", f"{rep['recent_ir']:.2f}")
    g3.metric("IR trend /yr", f"{rep['ir_trend_per_year']:+.2f}")
    g4.metric("Verdict", rep["verdict"])
    ir_series = monitoring.rolling_ir(net.dropna(), window=252).dropna()
    if len(ir_series):
        st.line_chart(ir_series, height=220)
    st.caption("The deflated Sharpe above is the gate at INCEPTION; this "
               "is the ongoing leg — rolling IR of the deployed signal. "
               "'DECAYING' (negative recent IR + negative trend) means "
               "retire or re-estimate, regardless of how good the "
               "original backtest was.")

    st.divider()
    st.subheader("View-conditioned allocation (entropy pooling)")
    mkt2, mkt_live2 = get_market()
    y10s = us_yields[10.0].resample("ME").last() \
        if 10.0 in us_yields.columns else None
    if y10s is not None:
        eq_m = (1 + mkt2["equity_ret"]).resample("ME").prod() - 1
        oas = mkt2["credit_spread_bp"].resample("ME").last()
        dy = y10s.diff().reindex(eq_m.index)
        govt_m = (y10s.reindex(eq_m.index) / 12 - 8.0 * dy) / 100
        cred_m = govt_m + (oas.reindex(eq_m.index) / 12
                           - 5.5 * oas.diff().reindex(eq_m.index)) / 1e4
        hist = pd.DataFrame({"govt": govt_m, "credit": cred_m,
                             "equity": eq_m}).dropna()
        rng = np.random.default_rng(5)
        scen = hist.to_numpy()[rng.integers(0, len(hist), 1000)]
        view_eq = st.slider("View: expected EQUITY return (% p.a.)",
                            -10.0, 15.0, 4.0, 0.5) / 100 / 12
        view_cr = st.slider("View: expected CREDIT return (% p.a.)",
                            -5.0, 10.0, 4.5, 0.5) / 100 / 12
        ep = al.EntropyPooling().fit(
            scen, np.vstack([al.view_on_mean(scen, 2),
                             al.view_on_mean(scen, 1)]),
            [view_eq, view_cr])
        mu_post, Sigma_post = ep.posterior_moments()
        ra = st.slider("Risk aversion", 1.0, 10.0, 4.0, 0.5)
        w = al.mv_optimize(mu_post, Sigma_post, risk_aversion=ra, w_max=0.7)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame(
                {"E[r] % p.a.": np.round(mu_post * 12 * 100, 2),
                 "vol % p.a.": np.round(np.sqrt(np.diag(Sigma_post) * 12)
                                        * 100, 2),
                 "weight": np.round(w, 3)}, index=hist.columns))
            st.metric("Effective scenarios (ENS)",
                      f"{ep.effective_n:,.0f} / 1,000")
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.bar(hist.columns, w, color=["#4477AA", "#EE6677", "#228833"])
            ax.set_ylabel("weight")
            ax.set_title("View-conditioned allocation")
            st.pyplot(fig, clear_figure=True)
            st.caption("Low ENS = the view is fighting the data — treat "
                       "the output weights with suspicion.")


# ====================================================== stress & resilience
with tabs[4]:
    st.caption("*Objective: forward-looking scenario analysis & stress "
               "testing across rates, spreads, FX and geopolitical risk — "
               "assess portfolio resilience, inform positioning.*")

    asia, as_ofs = get_asia_curves()
    jgb_c, _ = get_jgb_curve()
    ecb_c, _ = get_ecb_curve()

    st.subheader("Discounting basis")
    curve_opts = {"United States (UST, DNS-fitted)": ("dns", us_yields)}
    if jgb_c is not None:
        curve_opts["Japan (JGB, DNS-fitted)"] = ("dns", jgb_c)
    if ecb_c is not None:
        curve_opts["Euro area (AAA, DNS-fitted)"] = ("dns", ecb_c)
    for c in ("SG", "ID", "KR", "TH", "MY"):
        if c in asia:
            curve_opts[f"{ASIA_NAMES[c]} (ABO snapshot)"] = ("snap", asia[c])
    pick = st.selectbox("Liability discount curve", list(curve_opts))
    kind, obj = curve_opts[pick]
    if kind == "dns":
        mdl = fit_dns(obj)
        curve_fn = lambda t: ns_loadings(t, mdl.lam) \
            @ mdl.factors.iloc[-1].to_numpy() / 100
    else:
        curve_fn = data_global.curve_fn_from_snapshot(obj)

    st.subheader("Portfolio & liabilities")
    c = st.columns(5)
    mv = c[0].number_input("Assets MV (mn)", 1000.0, 1e6, 10_000.0,
                           step=500.0)
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

    gap = stress.duration_gap(pf, liab, curve_fn,
                              asset_dur=sum(pf.krd.values()))
    m = st.columns(4)
    m[0].metric("Liability PV (mn)", f"{gap['liab_pv']:,.0f}")
    m[1].metric("Duration gap (yrs)", f"{gap['dur_gap']:.2f}")
    m[2].metric("Economic surplus (mn)", f"{gap['surplus']:,.0f}")
    m[3].metric("Surplus Δ +100bp (mn)", f"{gap['surplus_chg_+100bp']:,.0f}")
    st.caption(f"Discounting on: {pick}. Same liability book discounted on "
               "a high-yield curve (ID) shows a smaller PV / duration than "
               "on JGBs — the cross-entity comparison AIM runs regionally.")

    col1, col2 = st.columns(2)
    col1.dataframe(stress.krd_gap(pf, liab, curve_fn))
    col2.dataframe(pd.Series(stress.capital_proxy(pf), name="value"))
    col2.caption("Capital proxy is illustrative only, NOT a regulatory "
                 "calculation.")

    st.subheader("Scenario library — market & geopolitical")
    scenarios = {
        "+100bp parallel": {"rates": {k: 100 for k in pf.krd}},
        "Bear steepener": {"rates": {2: 20, 5: 50, 10: 90, 20: 110, 30: 120}},
        "Fed shock +150bp": {"rates": {k: 150 for k in pf.krd},
                             "spreads_bp": 40, "equity_pct": -8,
                             "fx_pct": -4},
        "Taper tantrum '13": {"rates": {k: 80 for k in pf.krd},
                              "spreads_bp": 60, "equity_pct": -6,
                              "fx_pct": -8},
        "Credit blowout (GFC)": {"spreads_bp": 250, "equity_pct": -35,
                                 "rates": {k: -60 for k in pf.krd}},
        "Asia FX crisis '97-style": {"fx_pct": -20, "spreads_bp": 120,
                                     "equity_pct": -15,
                                     "rates": {2: 150, 5: 120, 10: 90,
                                               20: 70, 30: 60}},
        "China hard landing": {"rates": {k: -50 for k in pf.krd},
                               "spreads_bp": 180, "equity_pct": -28,
                               "fx_pct": -10},
        "Taiwan strait escalation": {"rates": {k: -40 for k in pf.krd},
                                     "spreads_bp": 150, "equity_pct": -25,
                                     "fx_pct": -12},
        "Oil supply shock": {"rates": {k: 60 for k in pf.krd},
                             "spreads_bp": 80, "equity_pct": -12,
                             "fx_pct": -6},
    }
    res = stress.run_scenarios(pf, scenarios)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    res[["rates", "spreads", "equity", "fx"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#4477AA", "#EE6677", "#228833", "#CCBB44"])
    ax.plot(range(len(res)), res["total"], "kD", label="total")
    ax.axhline(0, color="k", lw=0.7); ax.legend()
    ax.set_ylabel("P&L (mn)"); plt.xticks(rotation=18, ha="right")
    ax.set_title("Scenario P&L decomposition")
    st.pyplot(fig, clear_figure=True)
    st.dataframe(res)
    st.caption("Geopolitical scenarios are stylized multi-factor shocks "
               "(risk-off: rates rally, spreads/FX/equity sell off; "
               "inflationary: rates and spreads up together). Calibrate "
               "against history or an internal scenario committee before "
               "using for actual steering.")

    st.divider()
    st.subheader("Monte Carlo P&L distribution (1-month horizon)")
    mkt_mc, _ = get_market()
    fxr_mc = get_fx_all()
    if 10.0 in us_yields.columns and fxr_mc is not None \
            and "SGD" in fxr_mc.columns:
        y10m = us_yields[10.0]
        moves = pd.DataFrame({
            "rates_bp": y10m.diff() * 100,
            "spreads_bp": mkt_mc["credit_spread_bp"]
            .resample("ME").last().diff(),
            "equity_pct": ((1 + mkt_mc["equity_ret"])
                           .resample("ME").prod() - 1) * 100,
            "fx_pct": fxr_mc["SGD"].resample("ME").last()
            .pct_change() * 100,
        }).dropna()
        method_mc = st.radio("Simulation method",
                             ["bootstrap (historical rows, keeps tails)",
                              "normal (fitted mean/cov)"], horizontal=True)
        mc = stress.monte_carlo_pnl(
            pf, moves, n_sims=10_000,
            method="bootstrap" if method_mc.startswith("boot") else "normal")
        m = st.columns(5)
        m[0].metric("VaR 95 (mn)", f"{mc['var95']:,.0f}")
        m[1].metric("VaR 99 (mn)", f"{mc['var99']:,.0f}")
        m[2].metric("ES 95 (mn)", f"{mc['es95']:,.0f}")
        m[3].metric("ES 99 (mn)", f"{mc['es99']:,.0f}")
        m[4].metric("P(loss)", f"{mc['prob_loss']:.0%}")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.hist(mc["pnl"], bins=80, color="#4477AA", alpha=0.8)
        ax.axvline(-mc["var95"], color="orange", ls="--", label="VaR95")
        ax.axvline(-mc["var99"], color="red", ls="--", label="VaR99")
        ax.set_xlabel("1-month P&L (mn)"); ax.legend()
        st.pyplot(fig, clear_figure=True)
        st.caption(f"Calibrated on {len(moves)} historical monthly factor "
                   "moves (Δ10y UST, ΔIG OAS, equity return, USD/SGD move) "
                   "run through the SAME revaluation function as the "
                   "named scenarios. Bootstrap keeps the historical fat "
                   "tails and cross-factor dependence; the normal draws "
                   "understate them — the gap between the two ES99 numbers "
                   "IS the tail-risk story. Short spread history without a "
                   "FRED key (~3y) makes the tails optimistic.")


# ========================================================= manager oversight
with tabs[5]:
    st.caption("*Objective: support asset-manager oversight — evaluation, "
               "monitoring, performance assessment.*")
    st.caption(":blue[● SYNTHETIC] manager returns are simulated (no public "
               "manager data); the estimation machinery is real")
    rng = np.random.default_rng(11)
    T, n_mgr = 120, 20
    fac = pd.DataFrame(rng.normal(0.004, 0.03, (T, 2)),
                       columns=["mkt", "credit"],
                       index=pd.date_range("2016-01-31", periods=T,
                                           freq="ME"))
    true_alpha = np.zeros(n_mgr); true_alpha[[3, 11]] = 0.0025
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
                     "pval": round(out["pval"], 4),
                     "r2": round(out["r2"], 2)})
    panel = pd.DataFrame(rows).set_index("manager")
    bh = managers.benjamini_hochberg(panel["pval"], fdr=0.10)
    panel["BH significant (FDR 10%)"] = bh["significant_at_FDR"]
    col1, col2 = st.columns([3, 2])
    col1.dataframe(panel.sort_values("alpha_t", ascending=False), height=420)
    with col2:
        st.metric("Naive |t|>2 'skilled'", int((panel["alpha_t"].abs() > 2).sum()))
        st.metric("BH-FDR survivors",
                  int(panel["BH significant (FDR 10%)"].sum()))
        st.caption("2 truly skilled managers seeded (mgr_03, mgr_11). "
                   "Naive t-stat screens over-hire; FDR control is the fix.")
        pick = st.selectbox("Style-drift monitor", list(rets))
        rb = managers.rolling_betas(pd.Series(rets[pick], index=fac.index),
                                    fac, window=36)
        st.line_chart(rb, height=220)
