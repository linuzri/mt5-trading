"""
Supabase Sync Module
Pushes trading data to Supabase for cloud dashboard
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / '.env')
import requests
import json
from datetime import datetime, timezone
from typing import Optional
import threading
import queue

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://cxpablqwnwvacuvhcjen.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Headers for Supabase REST API
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# Async queue for non-blocking pushes
_push_queue = queue.Queue()
_worker_started = False


def _api_request(method: str, table: str, data: dict = None, params: str = ""):
    """Make a request to Supabase REST API"""
    url = f"{SUPABASE_URL}/rest/v1/{table}{params}"
    try:
        if method == "POST":
            resp = requests.post(url, headers=HEADERS, json=data, timeout=5)
        elif method == "PATCH":
            resp = requests.patch(url, headers=HEADERS, json=data, timeout=5)
        elif method == "GET":
            resp = requests.get(url, headers=HEADERS, timeout=5)
        elif method == "DELETE":
            resp = requests.delete(url, headers=HEADERS, timeout=5)
        else:
            return None
        
        if resp.status_code in [200, 201, 204]:
            return resp.json() if resp.text else True
        else:
            print(f"[SUPABASE] Error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"[SUPABASE] Request failed: {e}")
        return None


def _worker():
    """Background worker to process push queue"""
    while True:
        try:
            func, args, kwargs = _push_queue.get(timeout=60)
            func(*args, **kwargs)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[SUPABASE] Worker error: {e}")


def _ensure_worker():
    """Start background worker if not running"""
    global _worker_started
    if not _worker_started:
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        _worker_started = True


def _async_push(func, *args, **kwargs):
    """Queue a function for async execution"""
    _ensure_worker()
    _push_queue.put((func, args, kwargs))


# ============ Public API ============

def push_account_snapshot(balance: float, equity: float = None, initial_fund: float = 50000):
    """Push account balance snapshot"""
    def _push():
        data = {
            "balance": balance,
            "equity": equity or balance,
            "initial_fund": initial_fund
        }
        _api_request("POST", "account_snapshots", data)
    _async_push(_push)


def push_trade(bot_name: str, symbol: str, direction: str, entry_price: float, 
               exit_price: float, profit: float, confidence: float = None, source: str = "demo"):
    """Push completed trade to history"""
    def _push():
        data = {
            "bot_name": bot_name,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit": profit,
            "confidence": confidence,
            "source": source
        }
        _api_request("POST", "trades", data)
    _async_push(_push)


def update_bot_status(bot_name: str, status: str = "online", today_pnl: float = 0, 
                      today_trades: int = 0, today_wins: int = 0):
    """Update bot status (upsert)"""
    def _push():
        data = {
            "status": status,
            "today_pnl": today_pnl,
            "today_trades": today_trades,
            "today_wins": today_wins,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        # Use PATCH to update existing row
        url = f"{SUPABASE_URL}/rest/v1/bot_status?bot_name=eq.{bot_name}"
        try:
            resp = requests.patch(url, headers=HEADERS, json=data, timeout=5)
            if resp.status_code not in [200, 201, 204]:
                print(f"[SUPABASE] Bot status update failed: {resp.text[:200]}")
        except Exception as e:
            print(f"[SUPABASE] Bot status failed: {e}")
    _async_push(_push)


def push_log(bot_name: str, message: str):
    """Push log entry (keeps last 100 per bot)"""
    def _push():
        # Add new log
        data = {
            "bot_name": bot_name,
            "message": message[:500]  # Limit message length
        }
        _api_request("POST", "logs", data)
        
        # Clean old logs (keep last 100 per bot)
        # This runs occasionally, not every time
        import random
        if random.random() < 0.05:  # 5% chance to clean
            _cleanup_old_logs(bot_name)
    _async_push(_push)


def _cleanup_old_logs(bot_name: str):
    """Delete logs older than the most recent 100 for a bot"""
    try:
        # Get the 100th newest log ID for this bot
        url = f"{SUPABASE_URL}/rest/v1/logs?bot_name=eq.{bot_name}&order=id.desc&offset=100&limit=1&select=id"
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code == 200 and resp.json():
            cutoff_id = resp.json()[0]["id"]
            # Delete older logs
            delete_url = f"{SUPABASE_URL}/rest/v1/logs?bot_name=eq.{bot_name}&id=lt.{cutoff_id}"
            requests.delete(delete_url, headers=HEADERS, timeout=5)
    except Exception as e:
        pass  # Silent fail for cleanup


def update_daily_pnl(date_str: str, pnl: float, trades: int, wins: int):
    """Update daily P/L summary (upsert by checking if exists)"""
    def _push():
        # Check if date exists
        check_url = f"{SUPABASE_URL}/rest/v1/daily_pnl?date=eq.{date_str}&select=id"
        try:
            resp = requests.get(check_url, headers=HEADERS, timeout=5)
            data = {
                "date": date_str,
                "pnl": pnl,
                "trades": trades,
                "wins": wins
            }
            if resp.status_code == 200 and resp.json():
                # Update existing
                update_url = f"{SUPABASE_URL}/rest/v1/daily_pnl?date=eq.{date_str}"
                requests.patch(update_url, headers=HEADERS, json=data, timeout=5)
            else:
                # Insert new
                insert_url = f"{SUPABASE_URL}/rest/v1/daily_pnl"
                requests.post(insert_url, headers=HEADERS, json=data, timeout=5)
        except Exception as e:
            print(f"[SUPABASE] Daily P/L failed: {e}")
    _async_push(_push)


# ============ Test ============
if __name__ == "__main__":
    print("Testing Supabase connection...")
    
    # Test push
    push_account_snapshot(49500.00, 49500.00, 50000.00)
    update_bot_status("BTCUSD", "online", 150.50, 5, 3)
    push_log("BTCUSD", "[TEST] Supabase sync working!")
    
    # Wait for async pushes
    import time
    time.sleep(3)
    print("Done! Check Supabase dashboard for data.")


