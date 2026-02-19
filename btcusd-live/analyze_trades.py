#!/usr/bin/env python3
"""
Trade Performance Analyzer
Analyzes trade_log.csv to show trading statistics and performance metrics.

Usage:
    python analyze_trades.py              # Analyze all trades
    python analyze_trades.py --today      # Today's trades only
    python analyze_trades.py --days 7     # Last 7 days
    python analyze_trades.py --export     # Export detailed report to CSV
"""

import csv
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

def load_trades(filepath="trade_log.csv"):
    """Load trades from CSV file."""
    trades = []
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    try:
                        timestamp = datetime.fromisoformat(row[0].replace("+00:00", ""))
                        direction = row[1]
                        entry_price = float(row[2])
                        exit_price = float(row[3])
                        profit = float(row[4])
                        trades.append({
                            "timestamp": timestamp,
                            "direction": direction,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "profit": profit
                        })
                    except (ValueError, IndexError) as e:
                        continue
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return []
    return trades

def filter_trades_by_days(trades, days):
    """Filter trades to last N days."""
    cutoff = datetime.now() - timedelta(days=days)
    return [t for t in trades if t["timestamp"] >= cutoff]

def filter_trades_today(trades):
    """Filter to today's trades only."""
    today = datetime.now().date()
    return [t for t in trades if t["timestamp"].date() == today]

def analyze_trades(trades):
    """Calculate comprehensive trading statistics."""
    if not trades:
        return None

    # Basic counts
    total = len(trades)
    wins = [t for t in trades if t["profit"] > 0]
    losses = [t for t in trades if t["profit"] < 0]
    breakeven = [t for t in trades if t["profit"] == 0]

    # Direction breakdown
    buys = [t for t in trades if t["direction"] == "BUY"]
    sells = [t for t in trades if t["direction"] == "SELL"]

    buy_wins = len([t for t in buys if t["profit"] > 0])
    buy_losses = len([t for t in buys if t["profit"] < 0])
    sell_wins = len([t for t in sells if t["profit"] > 0])
    sell_losses = len([t for t in sells if t["profit"] < 0])

    # Profit calculations
    total_profit = sum(t["profit"] for t in wins) if wins else 0
    total_loss = abs(sum(t["profit"] for t in losses)) if losses else 0
    net_pnl = sum(t["profit"] for t in trades)

    avg_win = total_profit / len(wins) if wins else 0
    avg_loss = total_loss / len(losses) if losses else 0

    # Best/worst trades
    best_trade = max(trades, key=lambda t: t["profit"])
    worst_trade = min(trades, key=lambda t: t["profit"])

    # Win rate
    win_rate = (len(wins) / total * 100) if total > 0 else 0
    win_rate_excl_be = (len(wins) / (len(wins) + len(losses)) * 100) if (len(wins) + len(losses)) > 0 else 0

    # Profit factor
    profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')

    # Expectancy (average profit per trade)
    expectancy = net_pnl / total if total > 0 else 0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0

    for t in trades:
        if t["profit"] > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        elif t["profit"] < 0:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    # Time range
    first_trade = min(trades, key=lambda t: t["timestamp"])
    last_trade = max(trades, key=lambda t: t["timestamp"])

    # Daily breakdown
    daily_pnl = defaultdict(float)
    daily_trades = defaultdict(int)
    for t in trades:
        day = t["timestamp"].strftime("%Y-%m-%d")
        daily_pnl[day] += t["profit"]
        daily_trades[day] += 1

    profitable_days = sum(1 for pnl in daily_pnl.values() if pnl > 0)
    losing_days = sum(1 for pnl in daily_pnl.values() if pnl < 0)

    return {
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "buys": len(buys),
        "sells": len(sells),
        "buy_wins": buy_wins,
        "buy_losses": buy_losses,
        "sell_wins": sell_wins,
        "sell_losses": sell_losses,
        "total_profit": total_profit,
        "total_loss": total_loss,
        "net_pnl": net_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "win_rate": win_rate,
        "win_rate_excl_be": win_rate_excl_be,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "first_trade": first_trade,
        "last_trade": last_trade,
        "daily_pnl": dict(daily_pnl),
        "daily_trades": dict(daily_trades),
        "profitable_days": profitable_days,
        "losing_days": losing_days
    }

def print_report(stats):
    """Print formatted performance report."""
    if not stats:
        print("\n[!] No trades found to analyze.\n")
        return

    print("\n" + "=" * 60)
    print("           TRADE PERFORMANCE REPORT")
    print("=" * 60)

    # Time range
    print(f"\nPeriod: {stats['first_trade']['timestamp'].strftime('%Y-%m-%d %H:%M')} to {stats['last_trade']['timestamp'].strftime('%Y-%m-%d %H:%M')}")

    # Overall Summary
    print("\n" + "-" * 60)
    print("OVERALL SUMMARY")
    print("-" * 60)
    print(f"  Total Trades:        {stats['total']}")
    print(f"  Winning Trades:      {stats['wins']} ({stats['win_rate']:.1f}%)")
    print(f"  Losing Trades:       {stats['losses']} ({stats['losses']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "")
    print(f"  Breakeven Trades:    {stats['breakeven']} ({stats['breakeven']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "")
    print(f"  Win Rate (excl BE):  {stats['win_rate_excl_be']:.1f}%")

    # P/L Summary
    print("\n" + "-" * 60)
    print("PROFIT / LOSS")
    print("-" * 60)
    net_color = "+" if stats['net_pnl'] >= 0 else ""
    print(f"  Net P/L:             {net_color}${stats['net_pnl']:.2f}")
    print(f"  Total Profit:        +${stats['total_profit']:.2f}")
    print(f"  Total Loss:          -${stats['total_loss']:.2f}")
    print(f"  Profit Factor:       {stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "  Profit Factor:       N/A (no losses)")
    print(f"  Expectancy:          ${stats['expectancy']:.2f} per trade")

    # Averages
    print("\n" + "-" * 60)
    print("AVERAGES")
    print("-" * 60)
    print(f"  Average Win:         +${stats['avg_win']:.2f}")
    print(f"  Average Loss:        -${stats['avg_loss']:.2f}")
    if stats['avg_loss'] > 0:
        print(f"  Risk/Reward Ratio:   1:{stats['avg_win']/stats['avg_loss']:.2f}")

    # Best/Worst
    print("\n" + "-" * 60)
    print("BEST / WORST TRADES")
    print("-" * 60)
    best = stats['best_trade']
    worst = stats['worst_trade']
    print(f"  Best Trade:          +${best['profit']:.2f} ({best['direction']} @ {best['timestamp'].strftime('%Y-%m-%d %H:%M')})")
    print(f"  Worst Trade:         ${worst['profit']:.2f} ({worst['direction']} @ {worst['timestamp'].strftime('%Y-%m-%d %H:%M')})")

    # Direction Analysis
    print("\n" + "-" * 60)
    print("DIRECTION ANALYSIS")
    print("-" * 60)
    print(f"  BUY Trades:          {stats['buys']} (Wins: {stats['buy_wins']}, Losses: {stats['buy_losses']})")
    if stats['buys'] > 0:
        buy_wr = stats['buy_wins'] / (stats['buy_wins'] + stats['buy_losses']) * 100 if (stats['buy_wins'] + stats['buy_losses']) > 0 else 0
        print(f"    BUY Win Rate:      {buy_wr:.1f}%")
    print(f"  SELL Trades:         {stats['sells']} (Wins: {stats['sell_wins']}, Losses: {stats['sell_losses']})")
    if stats['sells'] > 0:
        sell_wr = stats['sell_wins'] / (stats['sell_wins'] + stats['sell_losses']) * 100 if (stats['sell_wins'] + stats['sell_losses']) > 0 else 0
        print(f"    SELL Win Rate:     {sell_wr:.1f}%")

    # Streaks
    print("\n" + "-" * 60)
    print("STREAKS")
    print("-" * 60)
    print(f"  Max Consecutive Wins:   {stats['max_consec_wins']}")
    print(f"  Max Consecutive Losses: {stats['max_consec_losses']}")

    # Daily Breakdown
    print("\n" + "-" * 60)
    print("DAILY BREAKDOWN")
    print("-" * 60)
    print(f"  Profitable Days:     {stats['profitable_days']}")
    print(f"  Losing Days:         {stats['losing_days']}")
    print("\n  Date          Trades    P/L")
    print("  " + "-" * 35)
    for day in sorted(stats['daily_pnl'].keys()):
        pnl = stats['daily_pnl'][day]
        count = stats['daily_trades'][day]
        sign = "+" if pnl >= 0 else ""
        print(f"  {day}    {count:3d}      {sign}${pnl:.2f}")

    # Recommendation
    print("\n" + "-" * 60)
    print("ASSESSMENT")
    print("-" * 60)

    if stats['win_rate_excl_be'] >= 50 and stats['profit_factor'] >= 1.5:
        print("  Status: GOOD - Strategy is profitable")
    elif stats['win_rate_excl_be'] >= 45 and stats['profit_factor'] >= 1.0:
        print("  Status: MARGINAL - Strategy is breaking even or slightly profitable")
    else:
        print("  Status: NEEDS IMPROVEMENT - Strategy is losing money")

    if stats['breakeven'] / stats['total'] > 0.3 if stats['total'] > 0 else False:
        print("  Warning: High breakeven rate (>30%) - trades reversing too quickly")

    if stats['avg_loss'] > stats['avg_win']:
        print("  Warning: Average loss > average win - consider tighter SL or wider TP")

    print("\n" + "=" * 60)
    print()

def export_report(stats, trades, filepath="trade_analysis_report.csv"):
    """Export detailed report to CSV."""
    if not stats:
        print("[!] No data to export.")
        return

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Summary section
        writer.writerow(["TRADE ANALYSIS REPORT"])
        writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])

        writer.writerow(["SUMMARY METRICS"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Trades", stats['total']])
        writer.writerow(["Wins", stats['wins']])
        writer.writerow(["Losses", stats['losses']])
        writer.writerow(["Breakeven", stats['breakeven']])
        writer.writerow(["Win Rate %", f"{stats['win_rate']:.2f}"])
        writer.writerow(["Win Rate (excl BE) %", f"{stats['win_rate_excl_be']:.2f}"])
        writer.writerow(["Net P/L", f"${stats['net_pnl']:.2f}"])
        writer.writerow(["Total Profit", f"${stats['total_profit']:.2f}"])
        writer.writerow(["Total Loss", f"${stats['total_loss']:.2f}"])
        writer.writerow(["Profit Factor", f"{stats['profit_factor']:.2f}"])
        writer.writerow(["Expectancy", f"${stats['expectancy']:.2f}"])
        writer.writerow(["Avg Win", f"${stats['avg_win']:.2f}"])
        writer.writerow(["Avg Loss", f"${stats['avg_loss']:.2f}"])
        writer.writerow([])

        # All trades
        writer.writerow(["ALL TRADES"])
        writer.writerow(["Timestamp", "Direction", "Entry Price", "Exit Price", "Profit", "Result"])
        for t in trades:
            result = "WIN" if t['profit'] > 0 else ("LOSS" if t['profit'] < 0 else "BE")
            writer.writerow([
                t['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                t['direction'],
                t['entry_price'],
                t['exit_price'],
                f"${t['profit']:.2f}",
                result
            ])

    print(f"\n[OK] Report exported to: {filepath}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze trading performance from trade_log.csv")
    parser.add_argument("--today", action="store_true", help="Analyze today's trades only")
    parser.add_argument("--days", type=int, help="Analyze last N days")
    parser.add_argument("--export", action="store_true", help="Export detailed report to CSV")
    parser.add_argument("--file", type=str, default="trade_log.csv", help="Path to trade log CSV")
    args = parser.parse_args()

    # Load trades
    trades = load_trades(args.file)

    if not trades:
        print("\n[!] No trades found in trade_log.csv\n")
        return

    # Filter if needed
    if args.today:
        trades = filter_trades_today(trades)
        print(f"\n[i] Filtering to today's trades...")
    elif args.days:
        trades = filter_trades_by_days(trades, args.days)
        print(f"\n[i] Filtering to last {args.days} days...")

    # Analyze
    stats = analyze_trades(trades)

    # Print report
    print_report(stats)

    # Export if requested
    if args.export:
        export_report(stats, trades)

if __name__ == "__main__":
    main()
