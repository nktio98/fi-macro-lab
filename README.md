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

### 1. `yield_curve.py` — Dynamic Nelson-Siegel engine
- Cross-sectional NS fit per date (lambda by grid search), VAR(1) factor
  dynamics, h-step curve forecasts. In-sample fit ~2-3bp RMSE.
- Upgrade path: AFNS yield-adjustment term (Christensen-Diebold-Rudebusch),
  ACM term-premium decomposition on the same factor panel, shadow-rate
  extension for near-ZLB markets (JPY, TWD).

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

### 10. `app.py` — Streamlit dashboard (live data)
- All modules interactive in the browser: `streamlit run app.py`.
- Deployable free on Streamlit Community Cloud (push repo to GitHub,
  point share.streamlit.io at `app.py`).

## Roadmap (extensions)
- MAS/SGS local-market data feeds; AFNS/ACM term premium; DCC-GARCH
  hedge ratios; BVAR scenario generator feeding entropy pooling;
  LLM-drafted daily commentary layer.
