"""
summary.py — Review overnight AutoResearch results
Run from btcusd/ directory: python summary.py
"""
import json
from pathlib import Path

# ── Param type labels — makes it easy to distinguish similar-looking params ───
PARAM_TYPE = {
    "sl_atr_multiplier":  "SL size",
    "tp_atr_multiplier":  "TP size",
    "min_atr":            "volatility filter",
    "h4_ema_fast":        "H4 trend fast EMA",
    "h4_ema_slow":        "H4 trend slow EMA",
    "h1_ema_period":      "H1 entry EMA",
    "atr_period":         "ATR period",
    "lot_size":           "position size",
}

def param_type(param: str) -> str:
    return PARAM_TYPE.get(param, param)

# ── Load log ──────────────────────────────────────────────────────────────────
log = Path("calibration.jsonl")
if not log.exists():
    print("No calibration.jsonl found.")
    exit()

exps = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
if not exps:
    print("Log is empty.")
    exit()

kept      = [e for e in exps if e["decision"] == "keep"]
discarded = [e for e in exps if e["decision"] == "discard"]
deployed  = [e for e in kept if e.get("deployed")]

# ── Header ────────────────────────────────────────────────────────────────────
W = 72
print("=" * W)
print(f"  AutoResearch Session Summary")
print("=" * W)
print(f"  Total experiments : {len(exps)}")
print(f"  Kept              : {len(kept)}")
print(f"  Deployed          : {len(deployed)}")
print(f"  Discarded         : {len(discarded)}")
if exps:
    print(f"  Keep rate         : {len(kept)/len(exps)*100:.0f}%")

# ── Best experiment ───────────────────────────────────────────────────────────
if kept:
    best = max(kept, key=lambda x: x["pnl"])
    dep  = " 🚀 deployed" if best.get("deployed") else ""
    print()
    print(f"  Best experiment{dep}:")
    print(f"    param     : {best['param']}  ({param_type(best['param'])})")
    print(f"    change    : {best['old_value']} -> {best['new_value']}")
    print(f"    win_rate  : {best['win_rate']}%")
    print(f"    pnl       : ${best['pnl']}")
    print(f"    drawdown  : {best['drawdown']}%")
    print(f"    rationale : {best.get('rationale', '')}")

# ── All kept ──────────────────────────────────────────────────────────────────
print()
print("  All kept experiments:")
print(f"  {'param':<22} {'type':<20} {'change':<14} {'WR%':>5} {'PnL':>8} {'DD%':>6}  {'deployed':>8}")
print("  " + "-" * (W - 2))
for e in kept:
    change = f"{e['old_value']} -> {e['new_value']}"
    dep    = "🚀 yes" if e.get("deployed") else ""
    print(
        f"  {e['param']:<22} {param_type(e['param']):<20} {change:<14} "
        f"{e['win_rate']:>5} {e['pnl']:>8} {e['drawdown']:>6}  {dep:>8}"
    )

# ── Last 5 discarded ──────────────────────────────────────────────────────────
if discarded:
    print()
    print("  Last 5 discarded:")
    print(f"  {'param':<22} {'type':<20} {'change':<14}  {'reason'}")
    print("  " + "-" * (W - 2))
    for e in discarded[-5:]:
        change = f"{e['old_value']} -> {e['new_value']}"
        reason = e.get("reason", "")[:38]
        print(f"  {e['param']:<22} {param_type(e['param']):<20} {change:<14}  {reason}")

# ── Footer ────────────────────────────────────────────────────────────────────
print("=" * W)
print("  Current params are in MEMORY.md (already updated).")
if deployed:
    print(f"  {len(deployed)} param(s) deployed to live bot.")
print("=" * W)
