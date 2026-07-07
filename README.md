# AIM Strategist Toolkit

Quant toolkit mirroring the core processes of an insurance investment
strategist role (Allianz Investment Management style): yield curve
modeling, regime detection, ALM-aware stress testing, FX analytics,
TAA validation, manager oversight, and view-conditioned allocation —
running on **live FRED data** with an interactive **Streamlit app**.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

pytest                        # validate estimators vs statsmodels (23 tests)
python run_real_demo.py       # live-data demo (FRED, no API key needed)
streamlit run app.py          # interactive dashboard on live data
```

Optional: set `FRED_API_KEY` (free at fred.stlouisfed.org) for
full-history credit spreads and equity; without it the public endpoint
caps BAML OAS at ~3y and SP500 at ~10y (Treasuries and FX are always
full-history). Live data is cached in `data_cache/` for 24h.

## Tests

`tests/test_validation.py` proves the from-scratch econometrics against
statsmodels references: ADF t-stat, Newey-West HAC standard errors,
Benjamini-Hochberg FDR, and VAR(1) coefficients all match to 1e-8.
`tests/test_internals.py` covers invariants and recovery on synthetic
data (regime accuracy, no-lookahead backtests, purged-CV leakage,
KRD additivity, entropy-pooling view attainment, BL limits).

## Modules

### 1. `yield_curve.py` — Dynamic Nelson-Siegel engine + AFNS + ACM
- Cross-sectional NS fit per date (lambda by grid search), VAR(1) factor
  dynamics, h-step curve forecasts. In-sample fit ~2-3bp RMSE.
- `AFNSModel`: arbitrage-free yield adjustment (Christensen-Diebold-
  Rudebusch closed form, two-step approximation) — convexity correction
  on the long end.
- `ACMTermPremium`: Adrian-Crump-Moench 3-step regression term premium;
  decomposes the 10y into expected short rates + term premium (identity
  tested exactly; pricing RMSE tested < 50bp).
- Remaining upgrade path: shadow-rate extension for near-ZLB markets.

### 2. `regimes.py` — Regime detection
- `GaussianMS`: 2-state Markov-switching (Hamilton filter + EM), from scratch.
- `JumpModel`: statistical jump model (k-means + switch penalty, exact DP
  state assignment) — the modern buy-side alternative; produces more
  persistent, more tradable regimes and takes arbitrary feature vectors
  (vol, momentum, spread changes, ...).

### 3. `stress.py` — ALM stress engine
- Portfolio via key-rate durations, spread duration, equity beta, FX delta.
- Stylized life liability book -> duration gap and economic surplus
  sensitivity (the insurance lens: falling rates HURT when liab dur > asset dur).
- Named scenarios (taper tantrum, credit blowout, Asia FX crisis) with P&L
  decomposition by risk factor.
- Illustrative Solvency-style market-risk capital aggregation (correlation
  matrix square-root rule). NOT a regulatory calculation.
- Upgrade path: BVAR / GARCH-DCC Monte Carlo feeding the same revaluation
  function; entropy pooling (Meucci) for view-conditioned distributions.

### `data.py`
- `load_yield_csv(path)`: drop in real data (FRED, MAS, Bloomberg export;
  first column date, remaining columns maturities in years, yields in %).
- Simulators used by the demo so everything runs offline.

## Run

```bash
python3 run_demo.py       # console report + charts in outputs/
```

### 4. `fx.py` — FX analytics (insurance investor lens)
- Hedge-cost engine: covered interest parity + cross-currency basis;
  hedged yield pickup decision table (hedged USD credit vs local bonds
  per investor currency) -- the core Asian insurance allocation question.
- Fair-value engine: Engle-Granger cointegration (from-scratch ADF test)
  + error-correction model -> misvaluation, half-life, +/-2sd signal bands.
- Rolling minimum-variance hedge ratio (upgrade path: DCC-GARCH betas).
- Demo: `python3 run_fx_demo.py`

### 5. `taa.py` — TAA research with anti-overfitting machinery
- Signal library (momentum, value z-score, carry) + z-score positioning.
- PurgedKFold cross-validation (purging + embargo, Lopez de Prado).
- Backtester net of transaction costs.
- Probabilistic & Deflated Sharpe ratios: strategies only pass if the
  Sharpe survives correction for non-normality AND number of trials.
- Demo: `python3 run_taa_demo.py`

### 6. `managers.py` — Asset manager oversight
- Factor regressions with Newey-West (HAC) alpha t-stats.
- Benjamini-Hochberg FDR control across the manager panel (the fix for
  "1-in-20 managers looks skilled by luck").
- Rolling-beta style-drift / mandate-compliance monitor; appraisal
  metrics (IR, tracking error, hit rate).
- Demo: `python3 run_manager_demo.py`

### 7. `allocation.py` — View-conditioned allocation
- Black-Litterman (reverse-optimized equilibrium + views).
- Entropy pooling (Meucci): impose views on a full scenario distribution
  by minimum relative entropy — handles non-normal stress-engine
  scenarios and views on any moment. Effective-scenario diagnostic.
- Constrained long-only mean-variance optimizer (SLSQP).
- Demo: `python3 run_allocation_demo.py`

### 8. `dashboard.py` — Self-contained HTML dashboard
- Single shareable .html embedding every chart and table; no server.
- Build everything end-to-end: `python3 build_dashboard.py`

### 9. `data_live.py` — Live data (FRED)
- US Treasury curve, Asian FX, US IG/HY OAS, S&P 500 + VIX; no API key
  required, optional `FRED_API_KEY` for full history; 24h disk cache.
- `python3 run_real_demo.py` — DNS + regimes + ALM stress on live data.

### 10. `data_global.py` — Multi-economy data (Asia + EU)
- AsianBondsOnline (ADB): live LCY government yield-curve snapshots for
  ASEAN+3 — China, Hong Kong, **Indonesia**, Japan, Korea, Malaysia,
  Philippines, **Singapore**, Thailand, Vietnam (+WTD/MTD/YTD changes).
- Japan MoF: full JGB curve history (daily, 1974→, 1Y-40Y).
- ECB: euro-area AAA spot-curve history; EUR reference FX (adds IDR).
- FRED OECD panel: monthly 10y yields for JP/KR/AU/DE/FR/IT/GB/US.
- All free, no API key; same 24h disk cache.

### 11. `nowcast.py` — Macro nowcasting (ID / KR / JP / US)
- Monthly activity factor via EM-PCA (missing-data tolerant — the core
  nowcasting trick for ragged publication lags).
- Bridge regression of quarterly GDP growth on the quarter-averaged
  factor (Newey-West inference) → current-quarter GDP nowcast.
- Honest about coverage: free monthly data exists for ID/KR/JP/US;
  SG/MY/TH have no usable free series (PMI is licensed).

### 12. `macro.py` — Shock transmission (local projections, Jordà 2005)
- Impulse responses of FX / spreads / equity to a US 10y shock, one OLS
  per horizon with Newey-West bands. Chosen over sign-restricted SVARs
  deliberately: same estimand, no identification controversy, auditable.

### 13. `monitoring.py` — Model & signal governance
- Forecast evaluation (hit rate, RMSE, IC), two-sided CUSUM drift
  detection on forecast errors, rolling-IR signal-decay reports.
- Complements `taa.deflated_sharpe` (the inception gate) with the
  post-deployment leg.

### Also upgraded
- `fx.py`: from-scratch GARCH(1,1) MLE + DCC(1,1) conditional
  minimum-variance hedge ratios (tested: recovers persistence and
  constant correlation on simulated data).
- `stress.py`: `monte_carlo_pnl` — bootstrap / normal Monte Carlo through
  the same revaluation function as the named scenarios → VaR/ES 95/99.

### 14. `app.py` — Streamlit dashboard (Asia-focused, live data)
- Organized around the strategist's three core objectives:
  1. *Economic & capital-market analysis* → *Rates & curves* (Asian curve
     monitor, DNS labs for UST/JGB/euro AAA, global 10y history) and
     *FX & regimes* (Asian FX monitor incl. IDR, regime detection on any
     series, ECM fair value with real rate differentials).
  2. *Translate views into TAA* → *Strategy & TAA* (hedged-pickup table
     computed from LIVE curves, purged-CV signal lab, entropy-pooling
     allocation).
  3. *Scenario analysis & stress testing* → *Stress & resilience*
     (liability discounting on any economy's curve — UST/JGB/euro/SGS/
     IndoGB —, KRD surplus gaps, market + geopolitical scenario library).
- Plus *Manager oversight* (synthetic panel, real machinery).
- Run: `streamlit run app.py`; deploy free on Streamlit Community Cloud.

## Roadmap (extensions)
- MAS/SGS local-market data feeds; AFNS/ACM term premium; DCC-GARCH
  hedge ratios; BVAR scenario generator feeding entropy pooling;
  LLM-drafted daily commentary layer.
