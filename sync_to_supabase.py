"""
Sync script: Import local trade history to Supabase (clears + reimports)
"""
import os
from dotenv import load_dotenv
load_dotenv()
import requests
import csv
from datetime import datetime
from collections import defaultdict

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://cxpablqwnwvacuvhcjen.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

TRADE_LOGS = {
    "BTCUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\btcusd\trade_log.csv",
    "XAUUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\xauusd\trade_log.csv",
    "EURUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\eurusd\trade_log.csv",
}

BATCH_SIZE = 50

def clear_table(table):
    """Delete all rows from a table"""
    # Supabase needs a filter for DELETE, use created_at > 1970
    resp = requests.delete(
        f"{SUPABASE_URL}/rest/v1/{table}?id=gt.0",
        headers=HEADERS,
        timeout=30
    )
    print(f"  Cleared {table}: {resp.status_code}")

def batch_insert(table, rows):
    """Insert rows in batches"""
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=HEADERS,
            json=batch,
            timeout=30
        )
        if resp.status_code in [200, 201, 204]:
            total += len(batch)
        else:
            print(f"  Batch error at {i}: {resp.text[:200]}")
    return total

def import_all():
    all_trades = []
    daily_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})

    for bot_name, csv_path in TRADE_LOGS.items():
        count = 0
        try:
            with open(csv_path, 'r') as f:
                for row in csv.reader(f):
                    if len(row) < 4:
                        continue
                    timestamp, direction, entry, exit_price = row[0], row[1], row[2], row[3]
                    try:
                        profit_val = float(row[4]) if len(row) > 4 and row[4] != 'N/A' else 0
                    except:
                        profit_val = 0

                    all_trades.append({
                        "created_at": timestamp,
                        "bot_name": bot_name,
                        "symbol": bot_name,
                        "direction": direction.upper(),
                        "entry_price": float(entry) if entry else None,
                        "exit_price": float(exit_price) if exit_price else None,
                        "profit": profit_val,
                    })
                    count += 1

                    try:
                        dt = datetime.fromisoformat(timestamp.replace('+00:00', ''))
                        date_str = dt.strftime('%Y-%m-%d')
                        daily_pnl[date_str]["pnl"] += profit_val
                        daily_pnl[date_str]["trades"] += 1
                        if profit_val > 0:
                            daily_pnl[date_str]["wins"] += 1
                    except:
                        pass

            print(f"  {bot_name}: {count} trades read")
        except FileNotFoundError:
            print(f"  {bot_name}: No trade log found (skipping)")
        except Exception as e:
            print(f"  {bot_name}: Error - {e}")

    # Clear and reimport trades
    print(f"\n[TRADES] Clearing old data...")
    clear_table("trades")
    print(f"[TRADES] Importing {len(all_trades)} trades in batches of {BATCH_SIZE}...")
    imported = batch_insert("trades", all_trades)
    print(f"[TRADES] Done: {imported} imported")

    # Clear and reimport daily P/L
    pnl_rows = [
        {"date": d, "pnl": round(s["pnl"], 2), "trades": s["trades"], "wins": s["wins"]}
        for d, s in sorted(daily_pnl.items())
    ]
    print(f"\n[DAILY P/L] Clearing old data...")
    clear_table("daily_pnl")
    print(f"[DAILY P/L] Importing {len(pnl_rows)} days...")
    days = batch_insert("daily_pnl", pnl_rows)
    print(f"[DAILY P/L] Done: {days} days imported")

    return imported, days

if __name__ == "__main__":
    print("=" * 50)
    print("Syncing local data to Supabase")
    print("=" * 50)
    trades, days = import_all()
    print(f"\n✅ Sync complete: {trades} trades, {days} days")


