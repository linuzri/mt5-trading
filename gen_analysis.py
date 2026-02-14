"""Generate daily analysis for a given date from trade CSVs and push to Supabase + JSON"""
import csv, os, sys, json, subprocess
from datetime import datetime, timedelta
from collections import defaultdict

date_str = sys.argv[1] if len(sys.argv) > 1 else (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

# Read trades from CSVs
bots = {'btcusd': 'btcusd/trade_log.csv', 'xauusd': 'xauusd/trade_log.csv', 'eurusd': 'eurusd/trade_log.csv'}
bot_pnl = defaultdict(float)
bot_trades = defaultdict(int)
total_pnl = 0
total_trades = 0

for bot, csv_path in bots.items():
    if not os.path.exists(csv_path):
        continue
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            ts = row[0]
            if date_str in ts:
                try:
                    pnl = float(row[4])
                    bot_pnl[bot] += pnl
                    bot_trades[bot] += 1
                    total_pnl += pnl
                    total_trades += 1
                except:
                    pass

print(f"Date: {date_str}")
print(f"Total trades: {total_trades}, P/L: ${total_pnl:.2f}")
for bot in ['btcusd', 'xauusd', 'eurusd']:
    print(f"  {bot}: {bot_trades[bot]} trades, ${bot_pnl[bot]:.2f}")

if total_trades == 0:
    print("No trades found for this date, skipping.")
    sys.exit(0)

# Get approximate balance from MT5
try:
    import MetaTrader5 as mt5
    mt5.initialize()
    info = mt5.account_info()
    balance = info.balance
    mt5.shutdown()
except:
    balance = 49220.0

# Build analysis
summary = f"Date: {date_str}. Trades: {total_trades}. P/L: ${total_pnl:.2f}."
for bot in ['btcusd', 'xauusd', 'eurusd']:
    if bot_trades[bot] > 0:
        summary += f" {bot.upper()}: {bot_trades[bot]} trades, ${bot_pnl[bot]:.2f}."

# Save to JSON
json_path = 'vercel-dashboard/data/daily_analysis.json'
data = []
if os.path.exists(json_path):
    with open(json_path) as f:
        data = json.load(f)

# Remove existing entry for this date
data = [d for d in data if d.get('date') != date_str]
entry = {
    'date': date_str,
    'balance': round(balance, 2),
    'total_pnl': round(total_pnl, 2),
    'total_trades': total_trades,
    'btcusd_pnl': round(bot_pnl['btcusd'], 2),
    'xauusd_pnl': round(bot_pnl['xauusd'], 2),
    'eurusd_pnl': round(bot_pnl['eurusd'], 2),
    'summary': summary
}
data.append(entry)
data.sort(key=lambda x: x['date'])

with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)
print(f"Saved to {json_path}")

# Push to Supabase
try:
    from dotenv import load_dotenv
    load_dotenv()
    import requests
    url = os.getenv("SUPABASE_URL", "https://cxpablqwnwvacuvhcjen.supabase.co")
    key = os.getenv("SUPABASE_KEY")
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
    r = requests.post(f"{url}/rest/v1/daily_analysis", headers=headers, json=entry)
    print(f"Supabase: {r.status_code} {r.text[:100]}")
except Exception as e:
    print(f"Supabase push failed: {e}")
