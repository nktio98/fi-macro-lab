"""
Live market data via FRED.

Two access paths:
  1. No API key (default): the public fredgraph.csv endpoint. History
     windows vary by series licence: Treasuries and FX are full-history,
     SP500 is capped at ~10y, BAML OAS spreads at ~3y.
  2. FRED_API_KEY set (free at https://fred.stlouisfed.org/docs/api/api_key.html):
     the official API is used instead and every series is full-history.

Everything is cached to ./data_cache/*.csv so repeated runs (and the
Streamlit app) don't hammer FRED and the toolkit still works offline
once a snapshot exists. Delete the cache folder to force a refresh.

Series used:
  US Treasury curve : DGS3MO..DGS30 (constant-maturity, daily, %)
  FX vs USD         : DEXSIUS (SGD), DEXJPUS (JPY), DEXKOUS (KRW),
                      DEXTHUS (THB), DEXMAUS (MYR), DEXCHUS (CNY)
  Credit spreads    : BAMLC0A0CM (US IG OAS), BAMLH0A0HYM2 (US HY OAS)
  Equity / vol      : SP500, VIXCLS
"""
from __future__ import annotations

import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
CACHE_DIR = Path("data_cache")
CACHE_MAX_AGE_HOURS = 24.0

TREASURY_SERIES = {  # FRED id -> maturity in years
    "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0,
    "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
}
FX_SERIES = {  # FRED id -> (name, quoted as)
    "DEXSIUS": ("SGD", "local_per_usd"), "DEXJPUS": ("JPY", "local_per_usd"),
    "DEXKOUS": ("KRW", "local_per_usd"), "DEXTHUS": ("THB", "local_per_usd"),
    "DEXMAUS": ("MYR", "local_per_usd"), "DEXCHUS": ("CNY", "local_per_usd"),
}


def _fetch_public(series_ids: list[str]) -> pd.DataFrame:
    r = requests.get(FRED_URL, params={"id": ",".join(series_ids)}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
    return df.apply(pd.to_numeric, errors="coerce")


def _fetch_api(series_ids: list[str], api_key: str,
               start: str = "1990-01-01") -> pd.DataFrame:
    cols = {}
    for sid in series_ids:
        r = requests.get(FRED_API_URL, timeout=60, params={
            "series_id": sid, "api_key": api_key, "file_type": "json",
            "observation_start": start})
        r.raise_for_status()
        obs = r.json()["observations"]
        s = pd.Series({o["date"]: o["value"] for o in obs}, name=sid)
        cols[sid] = pd.to_numeric(s, errors="coerce")
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    return df


def fred_csv(series_ids: list[str], start: str = "2000-01-01",
             cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch one or more FRED series as a date-indexed DataFrame (cached).
    Uses the official API (full history) when FRED_API_KEY is set."""
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    key = "_".join(sorted(series_ids))
    f = cache_dir / f"fred_{key}.csv"
    if f.exists() and (time.time() - f.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
    else:
        api_key = os.environ.get("FRED_API_KEY", "").strip()
        df = _fetch_api(series_ids, api_key) if api_key \
            else _fetch_public(series_ids)
        df.to_csv(f)
    return df.loc[df.index >= start]


def us_treasury_curve(start: str = "2000-01-01", monthly: bool = True,
                      **kw) -> pd.DataFrame:
    """US CMT yield panel, columns = maturities in years, values in %."""
    df = fred_csv(list(TREASURY_SERIES), start=start, **kw)
    df.columns = [TREASURY_SERIES[c] for c in df.columns]
    df = df.sort_index(axis=1).dropna(how="all")
    if monthly:
        df = df.resample("ME").last()
    return df.dropna()


def fx_rates(start: str = "2000-01-01", **kw) -> pd.DataFrame:
    """FX spot panel, columns = currency, values = LOCAL per USD."""
    df = fred_csv(list(FX_SERIES), start=start, **kw)
    df.columns = [FX_SERIES[c][0] for c in df.columns]
    return df.dropna(how="all").ffill()


def credit_spreads(start: str = "2000-01-01", **kw) -> pd.DataFrame:
    """US IG and HY option-adjusted spreads, in bp."""
    df = fred_csv(["BAMLC0A0CM", "BAMLH0A0HYM2"], start=start, **kw)
    df.columns = ["IG_OAS_bp", "HY_OAS_bp"]
    return (df * 100).dropna(how="all").ffill()


def equity_and_vix(start: str = "2015-01-01", **kw) -> pd.DataFrame:
    """S&P 500 level (FRED provides ~10y) and VIX."""
    df = fred_csv(["SP500", "VIXCLS"], start=start, **kw)
    df.columns = ["SP500", "VIX"]
    return df.dropna(how="all").ffill().dropna()


def market_snapshot(start: str = "2015-01-01", **kw) -> pd.DataFrame:
    """Daily panel for regime/TAA work: equity returns, spreads, VIX."""
    eq = equity_and_vix(start, **kw)
    cs = credit_spreads(start, **kw)
    out = pd.DataFrame({
        "equity_ret": eq["SP500"].pct_change(),
        "credit_spread_bp": cs["IG_OAS_bp"].reindex(eq.index).ffill(),
        "vix": eq["VIX"],
    }).dropna()
    return out
