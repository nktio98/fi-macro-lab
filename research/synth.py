"""
Synthetic bond panel with a PLANTED mispricing premium.

Used by the test suite (prove the machinery finds what is planted and
nothing when nothing is) and by the website's interactive methodology
demo (show the pipeline working without touching licensed data).

Pricing errors are AR(1)-persistent across months (phi=0.7), so the
planted premium has a realistic decay profile: cumulative-return
slopes grow with horizon at a decaying rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


RATING_NUM_MAP = {"AAA": 1, "AA": 4, "A": 6, "BBB": 9, "BB": 12, "B": 15}


def synth_panel(T: int = 90, n_bonds: int = 300, mis_premium: float = 0.08,
                phi: float = 0.7, downgrade_link: float = 0.0,
                seed: int = 9) -> pd.DataFrame:
    """downgrade_link > 0 plants an INFORMATION story: cheap bonds
    (positive pricing error) become more likely to be downgraded --
    the alternative hypothesis the downgrade-mechanism test must
    distinguish from pure mispricing (link = 0)."""
    r = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=T, freq="ME")
    dd = r.normal(7, 2, n_bonds)
    size = r.uniform(4, 9, n_bonds)
    dur = r.uniform(1, 12, n_bonds)
    tmt = np.clip(dur * r.uniform(1.0, 1.6, n_bonds), 0.5, 40)
    rating = r.choice(["AAA", "AA", "A", "BBB", "BB", "B"], n_bonds,
                      p=[0.05, 0.15, 0.3, 0.3, 0.12, 0.08])
    is_hy = np.isin(rating, ["BB", "B"])
    # AR(1) pricing errors, stationary std ~0.005
    innov_sd = 0.005 * np.sqrt(1 - phi ** 2)
    mis = np.zeros((T, n_bonds))
    mis[0] = r.normal(0, 0.005, n_bonds)
    for t in range(1, T):
        mis[t] = phi * mis[t - 1] + r.normal(0, innov_sd, n_bonds)
    # liquidity: persistent bond-level dollar volume (for double-sorts)
    dvol = np.exp(r.normal(6, 1.2, n_bonds))
    tmt_lab = pd.cut(tmt, bins=[0, 3, 7, 15, 100],
                     labels=["0-3y", "3-7y", "7-15y", ">15y"])
    R_next = np.empty((T, n_bonds))
    for t in range(T):
        R_next[t] = (0.003 + mis_premium * mis[t] * (~is_hy)
                     + r.normal(0, 0.006, n_bonds))
    # rating paths: baseline 1% monthly downgrade hazard, optionally
    # increasing in current cheapness (information story)
    rnum = np.empty((T, n_bonds))
    rnum[0] = np.array([RATING_NUM_MAP[c] for c in rating], dtype=float)
    for t in range(1, T):
        p_dg = 0.01 + downgrade_link * 40.0 * np.clip(mis[t - 1], 0, None)
        dg = r.uniform(size=n_bonds) < np.clip(p_dg, 0, 0.9)
        rnum[t] = np.minimum(rnum[t - 1] + dg, 22)
    rows = []
    for t in range(T):
        spread = (0.02 - 0.001 * dd - 0.002 * size + 0.0005 * dur
                  + 0.004 * is_hy + mis[t])
        ret_now = R_next[t - 1] if t else np.zeros(n_bonds)
        for i in range(n_bonds):
            rows.append({
                "ISSUE_ID": i, "DATE": dates[t], "PERMNO": 1000 + i,
                "T_Spread": spread[i], "RET_EOM": ret_now[i],
                "RET_EOM_next": R_next[t, i],
                "DDCamp": dd[i] + r.normal(0, 0.05),
                "log_size": size[i], "DURATION": dur[i],
                "AMOUNT_OUTSTANDING": np.exp(size[i]),
                "TMT": tmt[i], "TMT_bucket": tmt_lab[i],
                "T_DVolume": dvol[i] * r.uniform(0.5, 1.5),
                "RATING_CAT": rating[i],
                "RATING_NUM": rnum[t, i],
                "RATING_CLASS": "1.HY" if is_hy[i] else "0.IG"})
    return pd.DataFrame(rows)
