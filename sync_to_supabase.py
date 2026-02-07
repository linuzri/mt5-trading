"""
One-time sync script: Import local trade history to Supabase
"""
import requests
import csv
from datetime import datetime
from collections import defaultdict

SUPABASE_URL = "https://cxpablqwnwvacuvhcjen.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN4cGFibHF3bnd2YWN1dmhjamVuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA0MTQ2NTUsImV4cCI6MjA4NTk5MDY1NX0.IiA5SRPfoI9Y6ZaXTWcl4UmPwgzVJ7iBBRgXny4iGCE"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# Trade log files
TRADE_LOGS = {
    "BTCUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\\btcusd\trade_log.csv",
    "XAUUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\xauusd\trade_log.csv",
    "EURUSD": r"C:\Users\Nazri Hussain\projects\mt5-trading\eurusd\trade_log.csv",
}

daily_pnl = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})

def import_trades():
    """Import all trades from CSV files"""
    total = 0
    
    for bot_name, csv_path in TRADE_LOGS.items():
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 4:
                        continue
                    
                    timestamp, direction, entry, exit_price = row[0], row[1], row[2], row[3]
                    profit = row[4] if len(row) > 4 and row[4] != 'N/A' else None
                    
                    # Parse profit
                    try:
                        profit_val = float(profit) if profit else 0
                    except:
                        profit_val = 0
                    
                    # Parse date for daily P/L
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('+00:00', ''))
                        date_str = dt.strftime('%Y-%m-%d')
                        daily_pnl[date_str]["pnl"] += profit_val
                        daily_pnl[date_str]["trades"] += 1
                        if profit_val > 0:
                            daily_pnl[date_str]["wins"] += 1
                    except:
                        date_str = None
                    
                    # Push to Supabase
                    data = {
                        "created_at": timestamp,
                        "bot_name": bot_name,
                        "symbol": bot_name,
                        "direction": direction.upper(),
                        "entry_price": float(entry) if entry else None,
                        "exit_price": float(exit_price) if exit_price else None,
                        "profit": profit_val,
                    }
                    
                    resp = requests.post(
                        f"{SUPABASE_URL}/rest/v1/trades",
                        headers=HEADERS,
                        json=data,
                        timeout=10
                    )
                    
                    if resp.status_code in [200, 201, 204]:
                        total += 1
                    else:
                        print(f"  Error: {resp.text[:100]}")
                        
            print(f"[OK] {bot_name}: Imported trades")
        except FileNotFoundError:
            print(f"[SKIP] {bot_name}: No trade log found")
        except Exception as e:
            print(f"[ERROR] {bot_name}: {e}")
    
    return total

def import_daily_pnl():
    """Import daily P/L summaries"""
    count = 0
    
    for date_str, stats in sorted(daily_pnl.items()):
        data = {
            "date": date_str,
            "pnl": round(stats["pnl"], 2),
            "trades": stats["trades"],
            "wins": stats["wins"]
        }
        
        # Check if exists, then insert or update
        check = requests.get(
            f"{SUPABASE_URL}/rest/v1/daily_pnl?date=eq.{date_str}",
            headers=HEADERS,
            timeout=10
        )
        
        if check.status_code == 200 and check.json():
            # Update
            resp = requests.patch(
                f"{SUPABASE_URL}/rest/v1/daily_pnl?date=eq.{date_str}",
                headers=HEADERS,
                json=data,
                timeout=10
            )
        else:
            # Insert
            resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/daily_pnl",
                headers=HEADERS,
                json=data,
                timeout=10
            )
        
        if resp.status_code in [200, 201, 204]:
            count += 1
            print(f"  {date_str}: ${stats['pnl']:.2f} ({stats['trades']} trades)")
    
    return count

if __name__ == "__main__":
    print("=" * 50)
    print("Syncing local data to Supabase...")
    print("=" * 50)
    
    print("\n[TRADES] Importing trades...")
    trades = import_trades()
    print(f"Total trades imported: {trades}")
    
    print("\n[DAILY P/L] Importing daily P/L...")
    days = import_daily_pnl()
    print(f"Total days: {days}")
    
    print("\n[DONE] Sync complete!")
