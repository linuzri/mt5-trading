"""
Analyze: How many unanimous ensemble signals were blocked by the EMA trend filter?
What would have happened if we took those trades?
"""
import re
from datetime import datetime, timedelta, timezone
from collections import defaultdict

log_file = "btcusd/trade_notifications.log"

# Parse all log lines
with open(log_file, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Find blocked unanimous signals
blocked_signals = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # Look for unanimous ensemble signals (3/3 agree)
    if "3/3 agree" in line:
        # Extract timestamp, direction, ATR
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        dir_match = re.search(r"Ensemble: (BUY|SELL)", line)
        
        if ts_match and dir_match:
            ts_str = ts_match.group(1)
            direction = dir_match.group(1)
            
            # Check if next few lines have TREND FILTER blocked
            for j in range(i+1, min(i+5, len(lines))):
                if "TREND FILTER" in lines[j] and "blocked" in lines[j]:
                    blocked_signals.append({
                        "timestamp": ts_str,
                        "direction": direction,
                        "line": line.strip(),
                        "filter_line": lines[j].strip()
                    })
                    break
    i += 1

print(f"=== TREND FILTER BLOCKED UNANIMOUS SIGNALS ===")
print(f"Total blocked: {len(blocked_signals)}")
print()

# Group by date
by_date = defaultdict(list)
for s in blocked_signals:
    date = s["timestamp"][:10]
    by_date[date].append(s)

for date in sorted(by_date.keys()):
    signals = by_date[date]
    # Count unique 5-min windows (many signals per candle)
    windows = set()
    for s in signals:
        # Round to 5-min window
        ts = datetime.fromisoformat(s["timestamp"])
        minute_bucket = (ts.hour * 60 + ts.minute) // 5
        windows.add((s["direction"], minute_bucket))
    
    buy_blocked = sum(1 for d, _ in windows if d == "BUY")
    sell_blocked = sum(1 for d, _ in windows if d == "SELL")
    print(f"{date}: {len(windows)} unique signals blocked ({buy_blocked} BUY, {sell_blocked} SELL)")

print()

# Now find TAKEN unanimous signals (not blocked) for comparison
taken_signals = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if "3/3 agree" in line:
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        dir_match = re.search(r"Ensemble: (BUY|SELL)", line)
        
        if ts_match and dir_match:
            blocked = False
            for j in range(i+1, min(i+5, len(lines))):
                if "TREND FILTER" in lines[j] and "blocked" in lines[j]:
                    blocked = True
                    break
            if not blocked:
                # Check if a trade was actually placed
                for j in range(i+1, min(i+5, len(lines))):
                    if "NOTIFY" in lines[j] and "order placed" in lines[j]:
                        taken_signals.append({
                            "timestamp": ts_match.group(1),
                            "direction": dir_match.group(1)
                        })
                        break
    i += 1

print(f"=== TAKEN UNANIMOUS SIGNALS (not blocked) ===")
taken_windows = set()
for s in taken_signals:
    ts = datetime.fromisoformat(s["timestamp"])
    date = s["timestamp"][:10]
    minute_bucket = (ts.hour * 60 + ts.minute) // 5
    taken_windows.add((date, s["direction"], minute_bucket))

print(f"Total taken: {len(taken_windows)} unique signals")

# Now let's use MT5 to check what WOULD have happened on blocked signals
print()
print("=== BACKTESTING BLOCKED SIGNALS (using MT5 price data) ===")

try:
    import MetaTrader5 as mt5
    mt5.initialize()
    
    # Get config values
    sl_pips = 75  # from btcusd config
    tp_pips = 100
    lot = 0.05
    
    total_would_profit = 0
    total_would_loss = 0
    would_wins = 0
    would_losses = 0
    
    # Process each unique blocked signal window
    processed = set()
    for s in blocked_signals:
        ts = datetime.fromisoformat(s["timestamp"]).replace(tzinfo=timezone.utc)
        minute_bucket = (ts.hour * 60 + ts.minute) // 5
        key = (s["timestamp"][:10], s["direction"], minute_bucket)
        if key in processed:
            continue
        processed.add(key)
        
        # Get M1 candles after signal for fine-grained analysis
        rates = mt5.copy_rates_from("BTCUSD", mt5.TIMEFRAME_M1, ts, 120)  # 2 hours after
        if rates is None or len(rates) < 2:
            continue
        
        entry_price = rates[0][4]  # Close of signal candle
        direction = s["direction"]
        
        # Simulate trade: check if TP or SL hit first
        hit_tp = False
        hit_sl = False
        exit_price = entry_price
        hold_minutes = 0
        
        for r in rates[1:]:
            hold_minutes += 1
            if direction == "BUY":
                if r[2] >= entry_price + tp_pips:  # High hits TP
                    hit_tp = True
                    exit_price = entry_price + tp_pips
                    break
                if r[3] <= entry_price - sl_pips:  # Low hits SL
                    hit_sl = True
                    exit_price = entry_price - sl_pips
                    break
            else:  # SELL
                if r[3] <= entry_price - tp_pips:  # Low hits TP
                    hit_tp = True
                    exit_price = entry_price - tp_pips
                    break
                if r[2] >= entry_price + sl_pips:  # High hits SL
                    hit_sl = True
                    exit_price = entry_price + sl_pips
                    break
        
        # If neither hit in 2 hours, use last close
        if not hit_tp and not hit_sl:
            exit_price = rates[-1][4]
        
        if direction == "BUY":
            pnl = (exit_price - entry_price) * lot
        else:
            pnl = (entry_price - exit_price) * lot
        
        outcome = "TP" if hit_tp else "SL" if hit_sl else "TIMEOUT"
        if pnl > 0:
            would_wins += 1
            total_would_profit += pnl
        else:
            would_losses += 1
            total_would_loss += abs(pnl)
        
        date_str = s["timestamp"][:10]
        time_str = s["timestamp"][11:16]
        print(f"  {date_str} {time_str} {direction} @ {entry_price:.0f} -> {outcome} ({hold_minutes}min) P/L: ${pnl:.2f}")
    
    print()
    print("=== SUMMARY ===")
    total_blocked = would_wins + would_losses
    if total_blocked > 0:
        wr = would_wins / total_blocked * 100
        net = total_would_profit - total_would_loss
        print(f"Blocked signals backtested: {total_blocked}")
        print(f"Would-be wins: {would_wins} ({wr:.0f}%)")
        print(f"Would-be losses: {would_losses}")
        print(f"Would-be gross profit: ${total_would_profit:.2f}")
        print(f"Would-be gross loss: ${total_would_loss:.2f}")
        print(f"Would-be NET P/L: ${net:.2f}")
        print(f"Avg win: ${total_would_profit/max(would_wins,1):.2f}")
        print(f"Avg loss: ${total_would_loss/max(would_losses,1):.2f}")
        if net > 0:
            print(f"\n>>> VERDICT: Trend filter COST us ${net:.2f} by blocking these signals")
        else:
            print(f"\n>>> VERDICT: Trend filter SAVED us ${abs(net):.2f} by blocking these signals")
    
    mt5.shutdown()
except Exception as e:
    print(f"MT5 backtest failed: {e}")
    import traceback
    traceback.print_exc()
