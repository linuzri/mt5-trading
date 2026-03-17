"""
validate_variant.py — Extended backtest for Strategy Discovery variants
Usage: python validate_variant.py variants/bollinger_pullback_20260314_071846.py --hours 720
"""
import sys, json, argparse, importlib.util
from pathlib import Path

# Load backtest module
sys.path.insert(0, str(Path(__file__).parent))
import backtest_mt5 as bt

def load_variant_function(variant_path: str, func_name: str):
    """Load a function from a variant file, injecting module helpers."""
    import pandas as pd
    code = Path(variant_path).read_text(encoding="utf-8")
    namespace = {
        "pd": pd,
        "calc_ema": bt.calc_ema,
        "calc_atr": bt.calc_atr,
    }
    exec(compile(code, variant_path, "exec"), namespace)
    if func_name not in namespace:
        raise ValueError(f"Function '{func_name}' not found in {variant_path}")
    return namespace[func_name]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", help="Path to variant .py file")
    parser.add_argument("--hours", type=int, default=720, help="Backtest window hours")
    parser.add_argument("--func", default="replay_signals", help="Function to patch")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / "config.json"
    config = json.loads(config_path.read_text())
    params = {
        "sl_atr_mult": config.get("sl_atr_mult", 1.5),
        "tp_atr_mult": config.get("tp_atr_mult", 2.5),
        "h4_ema_fast": config.get("h4_ema_fast", 20),
        "h4_ema_slow": config.get("h4_ema_slow", 50),
        "h1_ema_period": config.get("h1_ema_period", 20),
        "atr_period": config.get("atr_period", 14),
        "min_atr": config.get("min_atr", 250),
    }

    print(f"=== VARIANT VALIDATION ===")
    print(f"Variant: {args.variant}")
    print(f"Window:  {args.hours}h ({args.hours/24:.0f} days)")
    print(f"Params:  SL={params['sl_atr_mult']} TP={params['tp_atr_mult']}")
    print()

    # Run baseline first
    print("--- BASELINE (original logic) ---")
    baseline = bt.run_backtest("btcusd", params, hours=args.hours)
    print(f"  WR={baseline['win_rate']:.1f}% | PnL=${baseline['pnl']:.2f} | "
          f"Trades={baseline['total_trades']} | DD={baseline['drawdown']:.2f}%")
    print()

    # Patch and run variant
    print(f"--- VARIANT ({Path(args.variant).stem}) ---")
    variant_func = load_variant_function(args.variant, args.func)
    original_func = getattr(bt, args.func)
    setattr(bt, args.func, variant_func)
    try:
        variant = bt.run_backtest("btcusd", params, hours=args.hours)
    finally:
        setattr(bt, args.func, original_func)

    print(f"  WR={variant['win_rate']:.1f}% | PnL=${variant['pnl']:.2f} | "
          f"Trades={variant['total_trades']} | DD={variant['drawdown']:.2f}%")
    print()

    # Comparison
    print("--- COMPARISON ---")
    wr_delta = variant["win_rate"] - baseline["win_rate"]
    pnl_delta = variant["pnl"] - baseline["pnl"]
    trades_delta = variant["total_trades"] - baseline["total_trades"]
    print(f"  Win Rate: {wr_delta:+.1f}%")
    print(f"  PnL:      ${pnl_delta:+.2f}")
    print(f"  Trades:   {trades_delta:+d}")
    print()

    if variant["win_rate"] >= baseline["win_rate"] * 0.95 and variant["pnl"] > baseline["pnl"]:
        print("✅ VARIANT PASSES extended validation")
    elif variant["pnl"] > baseline["pnl"] * 1.2:
        print("✅ VARIANT PASSES (PnL significantly better despite WR dip)")
    else:
        print("❌ VARIANT FAILS extended validation")

if __name__ == "__main__":
    main()
