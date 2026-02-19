"""Import missing trades from CSV to Supabase"""
import csv
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# Get existing live trades to avoid duplicates
resp = requests.get(
    f"{SUPABASE_URL}/rest/v1/trades?source=eq.live&select=created_at",
    headers={**HEADERS, "Prefer": ""},
    timeout=10
)
existing = set()
if resp.ok:
    for t in resp.json():
        existing.add(t['created_at'][:19])  # Compare up to seconds
    print(f"Found {len(existing)} existing live trades in Supabase")

# Read CSV and import missing
csv_file = Path(__file__).parent / 'trade_log.csv'
imported = 0
skipped = 0

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 5:
            continue
        ts, direction, entry, exit_p, profit = row[0], row[1], float(row[2]), float(row[3]), float(row[4])
        
        # Check if already exists
        if ts[:19] in existing:
            skipped += 1
            continue
        
        data = {
            "created_at": ts,
            "bot_name": "BTCUSD-LIVE",
            "symbol": "BTCUSD",
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_p,
            "profit": profit,
            "source": "live"
        }
        
        r = requests.post(f"{SUPABASE_URL}/rest/v1/trades", headers=HEADERS, json=data, timeout=5)
        if r.status_code in [200, 201, 204]:
            imported += 1
        else:
            print(f"Failed: {r.status_code} {r.text[:100]}")

print(f"Done! Imported: {imported}, Skipped (already exists): {skipped}")
