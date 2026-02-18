"""
Sync script: Incremental sync of local trade history to Supabase.
Only inserts trades that don't already exist (by bot_name + created_at).
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

# Also support Management API for schema changes
MGMT_TOKEN = os.environ.get("SUPABASE_MGMT_TOKEN", "")
MGMT_URL = "https://api.supabase.com/v1/projects/cxpablqwnwvacuvhcjen/database/query"
MGMT_HEADERS = {
    "Authorization": f"Bearer {MGMT_TOKEN}",
    "Content-Type": "application/json"
}

TRADE_LOGS = {
    "BTCUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\btcusd\trade_log.csv",
    "XAUUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\xauusd\trade_log.csv",
    "EURUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\eurusd\trade_log.csv",
    "BTCUSD-LIVE": r"C:\Users\Nazri Hussain\projects\mt5-trading\btcusd-live\trade_log.csv",
}

# Source mapping: which bots are live vs demo
BOT_SOURCE = {
    "BTCUSD": "demo",
    "XAUUSD": "demo",
    "EURUSD": "demo",
    "BTCUSD-LIVE": "live",
}

BATCH_SIZE = 50


def get_latest_trade_time(bot_name):
    """Get the most recent trade timestamp in Supabase for a bot"""
    url = f"{SUPABASE_URL}/rest/v1/trades?bot_name=eq.{bot_name}&select=created_at&order=created_at.desc&limit=1"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200 and resp.json():
            return resp.json()[0]["created_at"]
    except Exception as e:
        print(f"  Error getting latest time for {bot_name}: {e}")
    return None


def batch_upsert(table, rows):
    """Insert rows, skipping duplicates via ON CONFLICT"""
    headers = {**HEADERS, "Prefer": "resolution=ignore-duplicates,return=minimal"}
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=headers,
            json=batch,
            timeout=30
        )
        if resp.status_code in [200, 201, 204]:
            total += len(batch)
        else:
            print(f"  Batch error at {i}: {resp.text[:200]}")
    return total


def import_incremental():
    """Import only new trades (after the latest existing trade per bot)"""
    all_new_trades = []
    daily_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})

    for bot_name, csv_path in TRADE_LOGS.items():
        latest = get_latest_trade_time(bot_name)
        new_count = 0
        total_count = 0

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

                    total_count += 1

                    # Track daily P/L for ALL trades (needed for accurate daily totals)
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('+00:00', ''))
                        date_str = dt.strftime('%Y-%m-%d')
                        daily_pnl[date_str]["pnl"] += profit_val
                        daily_pnl[date_str]["trades"] += 1
                        if profit_val > 0:
                            daily_pnl[date_str]["wins"] += 1
                    except:
                        pass

                    # Only add trades newer than what's in Supabase
                    if latest and timestamp <= latest:
                        continue

                    all_new_trades.append({
                        "created_at": timestamp,
                        "bot_name": bot_name,
                        "symbol": bot_name.replace("-LIVE", ""),
                        "direction": direction.upper(),
                        "entry_price": float(entry) if entry else None,
                        "exit_price": float(exit_price) if exit_price else None,
                        "profit": profit_val,
                        "source": BOT_SOURCE.get(bot_name, "demo"),
                    })
                    new_count += 1

            print(f"  {bot_name}: {total_count} total, {new_count} new (latest in DB: {latest or 'none'})")
        except FileNotFoundError:
            print(f"  {bot_name}: No trade log found (skipping)")
        except Exception as e:
            print(f"  {bot_name}: Error - {e}")

    # Insert new trades
    imported = 0
    if all_new_trades:
        print(f"\n[TRADES] Inserting {len(all_new_trades)} new trades...")
        imported = batch_upsert("trades", all_new_trades)
        print(f"[TRADES] Done: {imported} inserted")
    else:
        print(f"\n[TRADES] No new trades to sync")

    # Upsert daily P/L (always update to get accurate totals)
    pnl_rows = [
        {"date": d, "pnl": round(s["pnl"], 2), "trades": s["trades"], "wins": s["wins"]}
        for d, s in sorted(daily_pnl.items())
    ]
    print(f"\n[DAILY P/L] Upserting {len(pnl_rows)} days...")
    # Use Management API for upsert with ON CONFLICT
    if MGMT_TOKEN:
        for row in pnl_rows:
            query = f"""INSERT INTO daily_pnl (date, pnl, trades, wins) 
VALUES ('{row["date"]}', {row["pnl"]}, {row["trades"]}, {row["wins"]})
ON CONFLICT (date) DO UPDATE SET pnl=EXCLUDED.pnl, trades=EXCLUDED.trades, wins=EXCLUDED.wins"""
            try:
                requests.post(MGMT_URL, headers=MGMT_HEADERS, json={"query": query}, timeout=10)
            except:
                pass
        print(f"[DAILY P/L] Done: {len(pnl_rows)} days upserted")
    else:
        # Fallback: clear and reimport
        resp = requests.delete(f"{SUPABASE_URL}/rest/v1/daily_pnl?id=gt.0", headers=HEADERS, timeout=30)
        days = batch_upsert("daily_pnl", pnl_rows)
        print(f"[DAILY P/L] Done: {days} days imported (full reimport)")

    return imported, len(pnl_rows)


def full_reimport():
    """Full clear and reimport (use with --full flag)"""
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
                        "symbol": bot_name.replace("-LIVE", ""),
                        "direction": direction.upper(),
                        "entry_price": float(entry) if entry else None,
                        "exit_price": float(exit_price) if exit_price else None,
                        "profit": profit_val,
                        "source": BOT_SOURCE.get(bot_name, "demo"),
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

    print(f"\n[TRADES] Clearing old data...")
    requests.delete(f"{SUPABASE_URL}/rest/v1/trades?id=gt.0", headers=HEADERS, timeout=30)
    print(f"[TRADES] Importing {len(all_trades)} trades...")
    imported = batch_upsert("trades", all_trades)
    print(f"[TRADES] Done: {imported} imported")

    pnl_rows = [
        {"date": d, "pnl": round(s["pnl"], 2), "trades": s["trades"], "wins": s["wins"]}
        for d, s in sorted(daily_pnl.items())
    ]
    requests.delete(f"{SUPABASE_URL}/rest/v1/daily_pnl?id=gt.0", headers=HEADERS, timeout=30)
    days = batch_upsert("daily_pnl", pnl_rows)
    print(f"[DAILY P/L] Done: {days} days imported")

    return imported, days


if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("Syncing local data to Supabase")
    print("=" * 50)

    if "--full" in sys.argv:
        print("Mode: FULL REIMPORT (clearing all data first)\n")
        trades, days = full_reimport()
    else:
        print("Mode: INCREMENTAL (only new trades)\n")
        trades, days = import_incremental()

    print(f"\nSync complete: {trades} new trades, {days} days P/L")
