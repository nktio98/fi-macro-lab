"""
Multi-economy market data: Asia (ASEAN+3), US, Japan, euro area.

Sources (all free, no API key):
  AsianBondsOnline (ADB)  : LCY government bond yield curve SNAPSHOTS
                            (latest close + WTD/MTD/YTD changes) for
                            CN, HK, ID, JP, KR, MY, PH, SG, TH, VN, US.
  Japan MoF               : full JGB curve HISTORY (daily, 1974->, 1Y-40Y).
  ECB Data Portal         : euro-area AAA spot curve HISTORY (2004->),
                            EUR reference FX rates (incl. IDR).
  FRED                    : US Treasury curve history (via data_live),
                            OECD 10y government yields (JP, KR, AU, EU4,
                            GB), Asian FX vs USD (via data_live).

Same disk cache convention as data_live (data_cache/, 24h TTL).
"""
from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data_live import CACHE_DIR, CACHE_MAX_AGE_HOURS, fred_csv

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

ABO_ECONOMIES = {  # code -> display name (curve endpoint serves ASEAN+3)
    "CN": "China", "HK": "Hong Kong", "ID": "Indonesia", "JP": "Japan",
    "KR": "Korea", "MY": "Malaysia", "PH": "Philippines", "SG": "Singapore",
    "TH": "Thailand", "VN": "Vietnam",
}

OECD_10Y = {  # FRED OECD monthly long-term government yields
    "JP": "IRLTLT01JPM156N", "KR": "IRLTLT01KRM156N",
    "AU": "IRLTLT01AUM156N", "DE": "IRLTLT01DEM156N",
    "FR": "IRLTLT01FRM156N", "IT": "IRLTLT01ITM156N",
    "GB": "IRLTLT01GBM156N", "US": "IRLTLT01USM156N",
}


def _cached_text(key: str, fetch_fn, max_age_h: float = CACHE_MAX_AGE_HOURS,
                 cache_dir: Path | None = None) -> str:
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    f = cache_dir / f"{key}.txt"
    if f.exists() and (time.time() - f.stat().st_mtime) < max_age_h * 3600:
        return f.read_text(encoding="utf-8")
    text = fetch_fn()
    f.write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------- AsianBondsOnline curves
def _parse_abo_tenor(label: str) -> float:
    m = re.match(r"([\d.]+)\s*(month|year)", label.strip().lower())
    if not m:
        raise ValueError(f"bad tenor {label!r}")
    v = float(m.group(1))
    return v / 12 if m.group(2) == "month" else v


def _parse_abo_xml(xml: str) -> pd.DataFrame:
    """Parse the ABO yield-curve XML into a tenor-indexed DataFrame."""
    as_of = ""
    m = re.search(r"Latest<br>Closing<br>\((\d{4}-\d{2}-\d{2})\)", xml)
    if m:
        as_of = m.group(1)
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
        # re.S on the cell regex too: CDATA content can contain \r or \n
        # (disk-cached copies get \r translated to \n by text mode)
        cells = re.findall(r"<cell[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</cell>",
                           row, re.S)
        if len(cells) >= 5:
            rows.append({"tenor": _parse_abo_tenor(cells[0]),
                         "yield_pct": float(cells[1]),
                         "wtd_bp": float(cells[2]),
                         "mtd_bp": float(cells[3]),
                         "ytd_bp": float(cells[4])})
    if rows:
        df = pd.DataFrame(rows).set_index("tenor").sort_index()
    else:
        df = pd.DataFrame(columns=["yield_pct", "wtd_bp", "mtd_bp", "ytd_bp"],
                          index=pd.Index([], name="tenor"))
    df.attrs["as_of"] = as_of
    return df


def abo_curve_snapshot(economy: str = "ID") -> pd.DataFrame:
    """Latest LCY government yield curve for one economy.

    Returns DataFrame indexed by tenor (years) with columns
    [yield_pct, wtd_bp, mtd_bp, ytd_bp] and .attrs['as_of'] date string."""
    def fetch():
        r = requests.get(
            "https://asianbondsonline.adb.org/xml/government_bond_yields_xml.php",
            params={"economy": economy}, timeout=60, headers=UA)
        r.raise_for_status()
        if "<cell" not in r.text:      # empty <rows/> = unsupported economy
            raise ValueError(f"no ABO data for economy={economy!r}")
        return r.text
    key = f"abo_curve_{economy}"
    df = _parse_abo_xml(_cached_text(key, fetch))
    if df.empty:                       # stale/poisoned cache -> drop & retry
        (CACHE_DIR / f"{key}.txt").unlink(missing_ok=True)
        df = _parse_abo_xml(_cached_text(key, fetch))
    if df.empty:
        raise ValueError(f"could not parse ABO curve for {economy!r}")
    df.attrs["economy"] = economy
    return df


def abo_curves(economies: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Curve snapshots for several economies; skips ones that fail."""
    out = {}
    for ec in (economies or list(ABO_ECONOMIES)):
        try:
            out[ec] = abo_curve_snapshot(ec)
        except Exception:
            continue
    return out


def curve_fn_from_snapshot(snap: pd.DataFrame):
    """Zero-curve function t(yrs) -> decimal yield, linearly interpolated
    from a snapshot curve (flat beyond the ends). For ALM discounting."""
    t = snap.index.to_numpy(dtype=float)
    y = snap["yield_pct"].to_numpy(dtype=float) / 100
    return lambda x: np.interp(np.asarray(x, dtype=float), t, y)


# ------------------------------------------------------- Japan MoF (JGB)
JGB_URL = ("https://www.mof.go.jp/english/policy/jgbs/reference/"
           "interest_rate/historical/jgbcme_all.csv")


def jgb_curve(start: str = "2000-01-01", monthly: bool = True) -> pd.DataFrame:
    """Full JGB yield curve history. Columns = maturities in years, %."""
    def fetch():
        r = requests.get(JGB_URL, timeout=120, headers=UA)
        r.raise_for_status()
        return r.text
    text = _cached_text("jgb_all", fetch)
    df = pd.read_csv(StringIO(text), skiprows=1)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
    df = df.set_index("Date")
    df.columns = [float(c.upper().replace("Y", "")) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[df.index >= start].dropna(axis=1, how="all")
    if monthly:
        df = df.resample("ME").last()
    return df.dropna()


# ------------------------------------------------------------ ECB curves
ECB_TENORS = {"3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
              "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "15Y": 15.0,
              "20Y": 20.0, "30Y": 30.0}


def ecb_curve(start: str = "2004-09-01", monthly: bool = True) -> pd.DataFrame:
    """Euro-area AAA government spot curve history (ECB Svensson fit)."""
    key = "+".join(f"SR_{k}" for k in ECB_TENORS)
    url = f"https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{key}"

    def fetch():
        r = requests.get(url, params={"format": "csvdata",
                                      "startPeriod": start},
                         timeout=120, headers=UA)
        r.raise_for_status()
        return r.text
    text = _cached_text("ecb_curve", fetch)
    df = pd.read_csv(StringIO(text))
    df = df.pivot_table(index="TIME_PERIOD", columns="DATA_TYPE_FM",
                        values="OBS_VALUE")
    df.index = pd.to_datetime(df.index)
    df.columns = [ECB_TENORS[c.replace("SR_", "")] for c in df.columns]
    df = df.sort_index(axis=1)
    if monthly:
        df = df.resample("ME").last()
    return df.dropna()


def ecb_fx_usd(currencies: list[str] = ("IDR", "INR", "TWD"),
               start: str = "2005-01-01") -> pd.DataFrame:
    """FX vs USD (local per USD) built from ECB EUR reference rates —
    covers currencies FRED's H.10 lacks (notably IDR)."""
    key = "+".join(list(currencies) + ["USD"])
    url = f"https://data-api.ecb.europa.eu/service/data/EXR/D.{key}.EUR.SP00.A"

    def fetch():
        r = requests.get(url, params={"format": "csvdata",
                                      "startPeriod": start},
                         timeout=120, headers=UA)
        r.raise_for_status()
        return r.text
    text = _cached_text(f"ecb_fx_{'_'.join(currencies)}", fetch)
    df = pd.read_csv(StringIO(text))
    df = df.pivot_table(index="TIME_PERIOD", columns="CURRENCY",
                        values="OBS_VALUE")
    df.index = pd.to_datetime(df.index)
    out = df[list(currencies)].div(df["USD"], axis=0)   # local/EUR ÷ USD/EUR
    return out.dropna(how="all").ffill()


# -------------------------------------------------- cross-country 10y panel
def global_10y_panel(start: str = "2000-01-01") -> pd.DataFrame:
    """Monthly 10y government yield history across major markets (FRED)."""
    df = fred_csv(list(OECD_10Y.values()), start=start)
    inv = {v: k for k, v in OECD_10Y.items()}
    df.columns = [inv[c] for c in df.columns]
    return df.dropna(how="all")
