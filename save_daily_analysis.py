#!/usr/bin/env python3
"""
Save daily analysis to the JSON file used by both dashboards.

Usage:
    python save_daily_analysis.py --date 2025-02-10 --balance 10200 --pnl 56 --trades 42 \
        --btcusd -12 --xauusd 68 --eurusd 0 --summary "Analysis text here..."

    # Or pipe summary from stdin:
    echo "Analysis text" | python save_daily_analysis.py --date 2025-02-10 --balance 10200 \
        --pnl 56 --trades 42 --btcusd -12 --xauusd 68 --eurusd 0

    # Or read summary from a file:
    python save_daily_analysis.py --date 2025-02-10 --balance 10200 --pnl 56 --trades 42 \
        --btcusd -12 --xauusd 68 --eurusd 0 --summary-file analysis.txt
"""

import argparse
import json
import sys
from pathlib import Path

JSON_PATH = Path(__file__).parent / "vercel-dashboard" / "data" / "daily_analysis.json"


def load_data():
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_data(data):
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {JSON_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Save daily trading analysis")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, required=True)
    parser.add_argument("--pnl", type=float, required=True, help="Total P/L")
    parser.add_argument("--trades", type=int, required=True, help="Total trades")
    parser.add_argument("--btcusd", type=float, default=0.0)
    parser.add_argument("--xauusd", type=float, default=0.0)
    parser.add_argument("--eurusd", type=float, default=0.0)
    parser.add_argument("--summary", default=None, help="Analysis text (markdown)")
    parser.add_argument("--summary-file", default=None, help="Read summary from file")
    args = parser.parse_args()

    # Get summary text
    if args.summary:
        summary = args.summary
    elif args.summary_file:
        summary = Path(args.summary_file).read_text(encoding="utf-8").strip()
    elif not sys.stdin.isatty():
        summary = sys.stdin.read().strip()
    else:
        print("Error: Provide --summary, --summary-file, or pipe text via stdin")
        sys.exit(1)

    entry = {
        "date": args.date,
        "summary": summary,
        "balance": args.balance,
        "total_pnl": args.pnl,
        "total_trades": args.trades,
        "btcusd_pnl": args.btcusd,
        "xauusd_pnl": args.xauusd,
        "eurusd_pnl": args.eurusd,
    }

    data = load_data()
    # Replace existing entry for same date, or append
    data = [d for d in data if d["date"] != args.date]
    data.append(entry)
    data.sort(key=lambda x: x["date"])
    save_data(data)
    print(f"Added analysis for {args.date}")


if __name__ == "__main__":
    main()
