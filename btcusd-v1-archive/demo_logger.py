"""
Structured CSV logging for demo week validation.
Three files: trades.csv, signals.csv, daily_summary.csv
All append-mode, headers written only on creation.
"""
import csv
import os
from datetime import datetime, timezone, timedelta

UTC = timezone.utc
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

TRADES_FILE = os.path.join(LOGS_DIR, "trades.csv")
SIGNALS_FILE = os.path.join(LOGS_DIR, "signals.csv")
DAILY_FILE = os.path.join(LOGS_DIR, "daily_summary.csv")

TRADES_HEADERS = [
    "timestamp_open", "timestamp_close", "direction", "entry_price", "exit_price",
    "sl_price", "tp_price", "pnl_dollars", "confidence", "rf_signal", "xgb_signal",
    "lgb_signal", "exit_reason", "bars_held", "atr_at_entry"
]

SIGNALS_HEADERS = [
    "timestamp", "signal", "confidence", "rf_signal", "xgb_signal", "lgb_signal",
    "executed", "blocked_by", "current_price", "atr"
]

DAILY_HEADERS = [
    "date", "trades_taken", "buys", "sells", "wins", "losses", "win_rate",
    "profit_factor", "daily_pnl", "signals_generated", "signals_blocked",
    "account_balance"
]


def _ensure_headers(filepath, headers):
    """Write headers if file doesn't exist or is empty."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def log_trade(timestamp_open, timestamp_close, direction, entry_price, exit_price,
              sl_price, tp_price, pnl_dollars, confidence, rf_signal, xgb_signal,
              lgb_signal, exit_reason, bars_held, atr_at_entry):
    """Append a closed trade to trades.csv"""
    try:
        _ensure_headers(TRADES_FILE, TRADES_HEADERS)
        with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                timestamp_open, timestamp_close, direction, 
                f"{entry_price:.2f}", f"{exit_price:.2f}",
                f"{sl_price:.2f}", f"{tp_price:.2f}", f"{pnl_dollars:.2f}",
                f"{confidence:.1%}" if isinstance(confidence, float) else confidence,
                rf_signal, xgb_signal, lgb_signal,
                exit_reason, bars_held, f"{atr_at_entry:.1f}" if atr_at_entry else ""
            ])
    except Exception as e:
        print(f"[DEMO LOG] Error writing trade: {e}")


def log_signal(signal, confidence, rf_signal, xgb_signal, lgb_signal,
               executed, blocked_by, current_price, atr):
    """Append an ML signal (executed or blocked) to signals.csv"""
    try:
        _ensure_headers(SIGNALS_FILE, SIGNALS_HEADERS)
        with open(SIGNALS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now(UTC).isoformat(),
                signal, 
                f"{confidence:.1%}" if isinstance(confidence, float) else confidence,
                rf_signal, xgb_signal, lgb_signal,
                executed, blocked_by,
                f"{current_price:.2f}" if current_price else "",
                f"{atr:.1f}" if atr else ""
            ])
    except Exception as e:
        print(f"[DEMO LOG] Error writing signal: {e}")


def log_daily_summary(date_str, trades_taken, buys, sells, wins, losses,
                      daily_pnl, signals_generated, signals_blocked, account_balance):
    """Append daily summary row to daily_summary.csv"""
    try:
        _ensure_headers(DAILY_FILE, DAILY_HEADERS)
        win_rate = f"{wins/trades_taken:.1%}" if trades_taken > 0 else "0.0%"
        gross_wins = daily_pnl if daily_pnl > 0 else 0  # Simplified
        gross_losses = abs(daily_pnl) if daily_pnl < 0 else 0
        profit_factor = f"{gross_wins/gross_losses:.2f}" if gross_losses > 0 else "inf" if gross_wins > 0 else "0.00"
        
        with open(DAILY_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                date_str, trades_taken, buys, sells, wins, losses,
                win_rate, profit_factor, f"{daily_pnl:.2f}",
                signals_generated, signals_blocked, f"{account_balance:.2f}"
            ])
    except Exception as e:
        print(f"[DEMO LOG] Error writing daily summary: {e}")


# Daily tracking state (in-memory, for computing daily summary)
_daily_state = {
    "date": None,
    "trades": 0,
    "buys": 0,
    "sells": 0,
    "wins": 0,
    "losses": 0,
    "pnl": 0.0,
    "signals": 0,
    "blocked": 0,
}


def track_signal(executed=True):
    """Increment signal counters for daily summary."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if _daily_state["date"] != today:
        # New day — flush previous day if it had data
        if _daily_state["date"] and _daily_state["signals"] > 0:
            flush_daily_summary(0.0)  # Balance will be 0 if we can't get it
        _daily_state.update({"date": today, "trades": 0, "buys": 0, "sells": 0,
                            "wins": 0, "losses": 0, "pnl": 0.0, "signals": 0, "blocked": 0})
    _daily_state["signals"] += 1
    if not executed:
        _daily_state["blocked"] += 1


def track_trade_close(direction, pnl, is_win):
    """Track a closed trade for daily summary."""
    _daily_state["trades"] += 1
    if direction.upper() == "BUY":
        _daily_state["buys"] += 1
    else:
        _daily_state["sells"] += 1
    if is_win:
        _daily_state["wins"] += 1
    else:
        _daily_state["losses"] += 1
    _daily_state["pnl"] += pnl


def flush_daily_summary(account_balance):
    """Write daily summary and reset. Call at midnight UTC or end of day."""
    s = _daily_state
    if s["date"] and (s["signals"] > 0 or s["trades"] > 0):
        log_daily_summary(
            s["date"], s["trades"], s["buys"], s["sells"],
            s["wins"], s["losses"], s["pnl"],
            s["signals"], s["blocked"], account_balance
        )
    # Reset
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    _daily_state.update({"date": today, "trades": 0, "buys": 0, "sells": 0,
                        "wins": 0, "losses": 0, "pnl": 0.0, "signals": 0, "blocked": 0})
