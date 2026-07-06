"""Tests for the multi-economy data layer. Parsing is tested offline with
canned payloads; live-endpoint smoke tests skip when the network is down."""
import numpy as np
import pytest

from aim_toolkit import data_global as dg

CANNED_XML = """<?xml version="1.0" encoding="iso-8859-1"?>
<rows>
  <head>
    <column><![CDATA[Tenor]]></column>
    <column><![CDATA[Latest<br>Closing<br>(2026-07-03)]]></column>
  </head>
  <row id="1"><cell><![CDATA[3 month]]></cell><cell>6.50</cell>
    <cell>1.0</cell><cell>2.0</cell><cell>3.0</cell></row>
  <row id="2"><cell><![CDATA[2 year]]></cell><cell>7.10</cell>
    <cell>-1.5</cell><cell>0.5</cell><cell>50.0</cell></row>
  <row id="3"><cell><![CDATA[10 year]]></cell><cell>7.21</cell>
    <cell>-3.2</cell><cell>-2.6</cell><cell>101.3</cell></row>
</rows>"""


def test_parse_abo_tenor():
    assert dg._parse_abo_tenor("3 month") == pytest.approx(0.25)
    assert dg._parse_abo_tenor("10 year") == 10.0
    assert dg._parse_abo_tenor(" 1 year\r") == 1.0
    with pytest.raises(ValueError):
        dg._parse_abo_tenor("overnight")


def test_parse_abo_xml():
    df = dg._parse_abo_xml(CANNED_XML)
    assert list(df.index) == [0.25, 2.0, 10.0]
    assert df.loc[10.0, "yield_pct"] == 7.21
    assert df.loc[2.0, "ytd_bp"] == 50.0
    assert df.attrs["as_of"] == "2026-07-03"


def test_parse_abo_xml_with_newlines_in_cdata():
    # Disk-cached copies get \r translated to \n inside CDATA (universal
    # newlines). The parser must survive both. Regression for a bug where
    # the cell regex lacked re.S and silently dropped every row.
    xml = CANNED_XML.replace("[3 month]", "[3 month\n]") \
                    .replace("[2 year]", "[2 year\r]")
    df = dg._parse_abo_xml(xml)
    assert list(df.index) == [0.25, 2.0, 10.0]


def test_parse_abo_xml_empty_rows_returns_empty_frame():
    df = dg._parse_abo_xml("<?xml version=\"1.0\"?><rows></rows>")
    assert df.empty and df.index.name == "tenor"


def test_curve_fn_from_snapshot_interpolates():
    df = dg._parse_abo_xml(CANNED_XML)
    fn = dg.curve_fn_from_snapshot(df)
    assert fn(10.0) == pytest.approx(0.0721)
    mid = fn(6.0)                          # between 2y (7.10) and 10y (7.21)
    assert 0.0710 < mid < 0.0721
    assert fn(30.0) == pytest.approx(0.0721)   # flat extrapolation
    assert fn(0.1) == pytest.approx(0.0650)


def _skip_if_offline(fn, name):
    try:
        return fn()
    except Exception as e:
        pytest.skip(f"{name} unreachable: {e}")


@pytest.mark.parametrize("economy", ["ID", "SG"])
def test_abo_curve_live(economy):
    snap = _skip_if_offline(lambda: dg.abo_curve_snapshot(economy), "ABO")
    assert len(snap) >= 5
    assert (snap["yield_pct"] > -2).all() and (snap["yield_pct"] < 25).all()
    assert snap.index.is_monotonic_increasing


def test_jgb_curve_live():
    jgb = _skip_if_offline(lambda: dg.jgb_curve(start="2020-01-01"), "MoF")
    assert 10.0 in jgb.columns and len(jgb) > 24
    assert np.isfinite(jgb.to_numpy()).all()


def test_ecb_curve_live():
    ecb = _skip_if_offline(lambda: dg.ecb_curve(), "ECB")
    assert 10.0 in ecb.columns and len(ecb) > 100


def test_ecb_fx_live():
    fxr = _skip_if_offline(lambda: dg.ecb_fx_usd(("IDR",)), "ECB FX")
    assert fxr["IDR"].dropna().iloc[-1] > 5000   # rupiah per USD
