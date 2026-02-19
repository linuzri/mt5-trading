"""
Backtest blocked unanimous signals WITH confidence threshold.
Test multiple thresholds to find the sweet spot.
"""
import re
from datetime import datetime, timedelta, timezone
from collections import defaultdict

log_file = "btcusd/trade_notifications.log"

with open(log_file, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Parse blocked unanimous signals WITH confidence data
blocked_signals = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    if "3/3 agree" in line:
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        dir_match = re.search(r"Ensemble: (BUY|SELL)", line)
        
        # Extract individual confidences: RF:SELL:62% XGB:SELL:70% LGB:SELL:52%
        conf_matches = re.findall(r"(?:RF|XGB|LGB):(?:BUY|SELL):(\d+)%", line)
        
        if ts_match and dir_match and len(conf_matches) == 3:
            confs = [int(c) for c in conf_matches]
            avg_conf = sum(confs) / 3
            
            # Check if blocked by trend filter
            for j in range(i+1, min(i+5, len(lines))):
                if "TREND FILTER" in lines[j] and "blocked" in lines[j]:
                    blocked_signals.append({
                        "timestamp": ts_match.group(1),
                        "direction": dir_match.group(1),
                        "confidences": confs,
                        "avg_conf": avg_conf,
                    })
                    break
    i += 1

print(f"Total blocked unanimous signals: {len(blocked_signals)}")

# Deduplicate by 5-min window
def dedup_signals(signals):
    seen = {}
    for s in signals:
        ts = datetime.fromisoformat(s["timestamp"])
        bucket = (s["timestamp"][:10], s["direction"], (ts.hour * 60 + ts.minute) // 5)
        if bucket not in seen:
            seen[bucket] = s
    return list(seen.values())

all_deduped = dedup_signals(blocked_signals)
print(f"Unique 5-min windows: {len(all_deduped)}")

# Backtest function
import MetaTrader5 as mt5
mt5.initialize()

sl_pips = 75
tp_pips = 100
lot = 0.05

def backtest_signals(signals, label):
    wins = 0
    losses = 0
    gross_profit = 0
    gross_loss = 0
    
    for s in signals:
        ts = datetime.fromisoformat(s["timestamp"]).replace(tzinfo=timezone.utc)
        rates = mt5.copy_rates_from("BTCUSD", mt5.TIMEFRAME_M1, ts, 120)
        if rates is None or len(rates) < 2:
            continue
        
        entry_price = rates[0][4]
        direction = s["direction"]
        
        hit_tp = False
        hit_sl = False
        exit_price = entry_price
        
        for r in rates[1:]:
            if direction == "BUY":
                if r[2] >= entry_price + tp_pips:
                    hit_tp = True
                    exit_price = entry_price + tp_pips
                    break
                if r[3] <= entry_price - sl_pips:
                    hit_sl = True
                    exit_price = entry_price - sl_pips
                    break
            else:
                if r[3] <= entry_price - tp_pips:
                    hit_tp = True
                    exit_price = entry_price - tp_pips
                    break
                if r[2] >= entry_price + sl_pips:
                    hit_sl = True
                    exit_price = entry_price + sl_pips
                    break
        
        if not hit_tp and not hit_sl:
            exit_price = rates[-1][4]
        
        if direction == "BUY":
            pnl = (exit_price - entry_price) * lot
        else:
            pnl = (entry_price - exit_price) * lot
        
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)
    
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    net = gross_profit - gross_loss
    return {
        "label": label,
        "total": total,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net": net,
    }

# Test multiple confidence thresholds
thresholds = [50, 55, 58, 60, 62, 65, 70]

print("\n" + "=" * 80)
print(f"{'Threshold':>10} | {'Signals':>8} | {'Wins':>5} | {'Losses':>6} | {'WR%':>6} | {'Net P/L':>10} | {'Per Trade':>10}")
print("-" * 80)

for thresh in thresholds:
    filtered = [s for s in all_deduped if s["avg_conf"] >= thresh]
    if not filtered:
        print(f"{thresh:>9}% | {'0':>8} | {'--':>5} | {'--':>6} | {'--':>6} | {'--':>10} | {'--':>10}")
        continue
    
    result = backtest_signals(filtered, f">={thresh}%")
    per_trade = result["net"] / result["total"] if result["total"] > 0 else 0
    print(f"{thresh:>9}% | {result['total']:>8} | {result['wins']:>5} | {result['losses']:>6} | {result['wr']:>5.1f}% | ${result['net']:>8.2f} | ${per_trade:>8.2f}")

# Also show confidence distribution
print("\n=== CONFIDENCE DISTRIBUTION (blocked signals) ===")
buckets = defaultdict(int)
for s in all_deduped:
    bucket = int(s["avg_conf"] // 5) * 5
    buckets[bucket] += 1

for b in sorted(buckets.keys()):
    bar = "#" * (buckets[b] // 2)
    print(f"  {b}-{b+4}%: {buckets[b]:>4} signals {bar}")

# Show what today's signal looked like
print("\n=== TODAY'S BLOCKED SIGNALS (for reference) ===")
today_signals = [s for s in all_deduped if s["timestamp"].startswith("2026-02-17")]
for s in today_signals:
    print(f"  {s['timestamp'][11:16]} {s['direction']} avg:{s['avg_conf']:.1f}% ({s['confidences']})")

mt5.shutdown()
