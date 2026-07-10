"""Run the research extensions on the real panel and save artifacts.

Usage: python -m research.run_extensions [--mode window]
"""
from __future__ import annotations

import argparse
import json

from .artifacts import ARTIFACT_DIR
from .extensions import decay_profile, liquidity_double_sort, \
    regime_conditional_fmb, strategy_turnover
from .panel import build_panel
from .signals import spread_residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "window"], default="window")
    args = ap.parse_args()
    print(f"building panel (mode={args.mode})...")
    panel = build_panel(mode=args.mode)
    p = spread_residuals(panel)["panel"]
    out = ARTIFACT_DIR / args.mode / "extensions"
    out.mkdir(parents=True, exist_ok=True)

    print("1/3 decay profile...")
    decay = decay_profile(p)
    decay.round(6).to_csv(out / "decay_profile.csv")
    print(decay.round(4).to_string())

    print("2/3 liquidity double sort + turnover...")
    liq = liquidity_double_sort(p)
    liq.round(6).to_csv(out / "liquidity_double_sort.csv")
    print(liq.round(4).to_string())
    turn = strategy_turnover(p)
    (out / "turnover.json").write_text(json.dumps(
        {k: round(v, 6) for k, v in turn.items()}, indent=2),
        encoding="utf-8")
    print(f"   turnover={turn['one_way_turnover']:.2f}, "
          f"breakeven={turn['breakeven_cost_bp']:.1f}bp one-way")

    print("3/3 regime-conditional FMB...")
    reg = regime_conditional_fmb(p)
    reg["table"].round(6).to_csv(out / "regime_conditional.csv")
    reg["regime"].to_csv(out / "regime_series.csv")
    print(reg["table"].round(4).to_string())
    print(f"   stress share={reg['stress_share']:.1%}, "
          f"diff t (stress-calm)={reg['diff_t_stress_minus_calm']:.2f}")
    print(f"artifacts -> {out}")


if __name__ == "__main__":
    main()
