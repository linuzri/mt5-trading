#!/usr/bin/env python3
"""Push daily analysis from JSON to Supabase. Usage: python push_analysis_supabase.py [--date YYYY-MM-DD]"""
import json, sys, argparse, requests
from pathlib import Path
from datetime import datetime

JSON_PATH = Path(__file__).parent / "vercel-dashboard" / "data" / "daily_analysis.json"
API_URL = "https://api.supabase.com/v1/projects/cxpablqwnwvacuvhcjen/database/query"

# Load token from .env
env_path = Path(__file__).parent / ".env"
token = None
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("SUPABASE_MGMT_TOKEN="):
            token = line.split("=", 1)[1].strip()
if not token:
    print("Error: SUPABASE_MGMT_TOKEN not found in .env")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--date", default=None, help="Date to push (default: latest)")
args = parser.parse_args()

data = json.load(open(JSON_PATH, encoding="utf-8"))
if args.date:
    entries = [d for d in data if d["date"] == args.date]
else:
    entries = [data[-1]] if data else []

if not entries:
    print(f"No entry found for {args.date or 'latest'}")
    sys.exit(1)

for entry in entries:
    summary_escaped = entry["summary"].replace("'", "''")
    query = f"""INSERT INTO daily_analysis (date, summary, balance, total_pnl, total_trades, btcusd_pnl, xauusd_pnl, eurusd_pnl)
VALUES ('{entry["date"]}', '{summary_escaped}', {entry["balance"]}, {entry["total_pnl"]}, {entry["total_trades"]}, {entry.get("btcusd_pnl", 0)}, {entry.get("xauusd_pnl", 0)}, {entry.get("eurusd_pnl", 0)})
ON CONFLICT (date) DO UPDATE SET summary=EXCLUDED.summary, balance=EXCLUDED.balance, total_pnl=EXCLUDED.total_pnl, total_trades=EXCLUDED.total_trades, btcusd_pnl=EXCLUDED.btcusd_pnl, xauusd_pnl=EXCLUDED.xauusd_pnl, eurusd_pnl=EXCLUDED.eurusd_pnl"""

    r = requests.post(API_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"query": query})
    print(f"{entry['date']}: {r.status_code} {r.json()}")
