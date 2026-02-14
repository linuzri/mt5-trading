"""
XAUUSD Backtest: OLD config vs NEW config (Feb 14, 2026 changes)
Simulates how new safety limits would have affected historical trades.
"""

import csv
from datetime import datetime, timedelta
from collections import defaultdict

TRADE_LOG = "xauusd/trade_log.csv"
OUTPUT_FILE = "xauusd_backtest_results.txt"

# Config changes
OLD_LOT = 0.05
NEW_LOT = 0.02
LOT_RATIO = NEW_LOT / OLD_LOT  # 0.4

MAX_DAILY_LOSS = -200.0
MAX_CONSECUTIVE_LOSSES = 5
CIRCUIT_BREAKER_HOURS = 1
LOSS_COOLDOWN_MIN = 10


def parse_trades(path):
    trades = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 5:
                continue
            ts_str, direction, entry, exit_p, profit = row[0], row[1], float(row[2]), float(row[3]), float(row[4])
            ts = datetime.fromisoformat(ts_str)
            trades.append({
                "ts": ts,
                "direction": direction,
                "entry": entry,
                "exit": exit_p,
                "profit_old": profit,
                "profit_new": profit * LOT_RATIO,
            })
    return trades


def get_date_key(ts):
    return ts.strftime("%Y-%m-%d")


def simulate_old(trades):
    """Old config: no limits, just raw P/L."""
    daily = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "losses_list": []})
    for t in trades:
        d = get_date_key(t["ts"])
        daily[d]["trades"] += 1
        daily[d]["pnl"] += t["profit_old"]
        if t["profit_old"] > 0:
            daily[d]["wins"] += 1
        daily[d]["losses_list"].append(t["profit_old"])
    return daily


def simulate_new(trades):
    """New config with all safety limits applied."""
    daily = defaultdict(lambda: {"trades": 0, "skipped": 0, "wins": 0, "pnl": 0.0,
                                  "skip_reasons": defaultdict(int)})
    
    consecutive_losses = 0
    circuit_breaker_until = None
    last_loss_time = None
    current_day = None
    daily_loss_accum = 0.0
    daily_loss_capped = False

    for t in trades:
        d = get_date_key(t["ts"])
        pnl = t["profit_new"]

        # Reset daily tracking on new day
        if d != current_day:
            current_day = d
            daily_loss_accum = 0.0
            daily_loss_capped = False

        # Check daily loss cap
        if daily_loss_capped:
            daily[d]["skipped"] += 1
            daily[d]["skip_reasons"]["daily_loss_cap"] += 1
            continue

        # Check circuit breaker (5 consecutive losses -> skip 1 hour)
        if circuit_breaker_until and t["ts"] < circuit_breaker_until:
            daily[d]["skipped"] += 1
            daily[d]["skip_reasons"]["circuit_breaker"] += 1
            continue

        # Check 10-min cooldown after loss
        if last_loss_time and t["ts"] < last_loss_time + timedelta(minutes=LOSS_COOLDOWN_MIN):
            daily[d]["skipped"] += 1
            daily[d]["skip_reasons"]["cooldown"] += 1
            continue

        # Trade is taken
        daily[d]["trades"] += 1
        daily[d]["pnl"] += pnl

        if pnl > 0:
            daily[d]["wins"] += 1
            consecutive_losses = 0
        elif pnl < 0:
            consecutive_losses += 1
            last_loss_time = t["ts"]
            daily_loss_accum += pnl

            if daily_loss_accum <= MAX_DAILY_LOSS:
                daily_loss_capped = True

            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                circuit_breaker_until = t["ts"] + timedelta(hours=CIRCUIT_BREAKER_HOURS)
                consecutive_losses = 0
        else:
            # breakeven
            pass

    return daily


def compute_drawdown(trades, use_new=False):
    """Compute max drawdown from equity curve."""
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        pnl = t["profit_new"] if use_new else t["profit_old"]
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main():
    trades = parse_trades(TRADE_LOG)
    old_daily = simulate_old(trades)
    new_daily = simulate_new(trades)

    all_dates = sorted(set(list(old_daily.keys()) + list(new_daily.keys())))

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("XAUUSD BACKTEST: OLD CONFIG vs NEW CONFIG")
    out("=" * 100)
    out(f"Period: {all_dates[0]} to {all_dates[-1]}  |  Total trades in log: {len(trades)}")
    out(f"Lot size: {OLD_LOT} -> {NEW_LOT}  |  Daily loss cap: None -> ${abs(MAX_DAILY_LOSS)}")
    out(f"Max consecutive losses: None -> {MAX_CONSECUTIVE_LOSSES}  |  Cooldown: 5min -> {LOSS_COOLDOWN_MIN}min")
    out()

    # Daily comparison table
    header = f"{'Date':<12} | {'OLD Trades':>10} {'OLD Win%':>8} {'OLD P/L':>10} | {'NEW Trades':>10} {'NEW Skip':>8} {'NEW Win%':>8} {'NEW P/L':>10} | {'Saved':>10}"
    out(header)
    out("-" * len(header))

    total_old_pnl = 0.0
    total_new_pnl = 0.0
    total_old_trades = 0
    total_new_trades = 0
    total_new_skipped = 0
    total_old_wins = 0
    total_new_wins = 0

    for d in all_dates:
        o = old_daily[d]
        n = new_daily[d]

        o_wr = (o["wins"] / o["trades"] * 100) if o["trades"] > 0 else 0
        n_wr = (n["wins"] / n["trades"] * 100) if n["trades"] > 0 else 0
        saved = n["pnl"] - o["pnl"]

        out(f"{d:<12} | {o['trades']:>10} {o_wr:>7.1f}% {o['pnl']:>+10.2f} | {n['trades']:>10} {n.get('skipped',0):>8} {n_wr:>7.1f}% {n['pnl']:>+10.2f} | {saved:>+10.2f}")

        total_old_pnl += o["pnl"]
        total_new_pnl += n["pnl"]
        total_old_trades += o["trades"]
        total_new_trades += n["trades"]
        total_new_skipped += n.get("skipped", 0)
        total_old_wins += o["wins"]
        total_new_wins += n["wins"]

    out("-" * len(header))
    old_wr = (total_old_wins / total_old_trades * 100) if total_old_trades > 0 else 0
    new_wr = (total_new_wins / total_new_trades * 100) if total_new_trades > 0 else 0
    total_saved = total_new_pnl - total_old_pnl
    out(f"{'TOTAL':<12} | {total_old_trades:>10} {old_wr:>7.1f}% {total_old_pnl:>+10.2f} | {total_new_trades:>10} {total_new_skipped:>8} {new_wr:>7.1f}% {total_new_pnl:>+10.2f} | {total_saved:>+10.2f}")

    out()
    out("=" * 60)
    out("SUMMARY")
    out("=" * 60)
    out(f"  OLD Config Total P/L:   ${total_old_pnl:>+10.2f}")
    out(f"  NEW Config Total P/L:   ${total_new_pnl:>+10.2f}")
    out(f"  Difference (saved):     ${total_saved:>+10.2f}")
    out()
    out(f"  OLD Trades Taken:       {total_old_trades}")
    out(f"  NEW Trades Taken:       {total_new_trades}")
    out(f"  NEW Trades Skipped:     {total_new_skipped}")
    out()
    out(f"  OLD Win Rate:           {old_wr:.1f}%")
    out(f"  NEW Win Rate:           {new_wr:.1f}%")
    out()

    # Max drawdown
    old_dd = compute_drawdown(trades, use_new=False)
    new_dd = compute_drawdown(trades, use_new=True)
    out(f"  OLD Max Drawdown:       ${old_dd:>10.2f}")
    out(f"  NEW Max Drawdown:       ${new_dd:>10.2f}")
    out(f"  Drawdown Reduction:     ${old_dd - new_dd:>10.2f}")
    out()

    # Skip reason breakdown
    out("SKIP REASONS (NEW config):")
    reason_totals = defaultdict(int)
    for d in all_dates:
        n = new_daily[d]
        for reason, count in n.get("skip_reasons", {}).items():
            reason_totals[reason] += count
    for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
        out(f"  {reason:<20}: {count} trades skipped")

    out()
    out("NOTE: NEW P/L = OLD P/L * 0.4 (lot size reduction) + safety limit filtering")
    out("      Max drawdown is computed on raw equity curve (no daily reset)")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
