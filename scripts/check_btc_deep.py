import csv
from datetime import datetime
from collections import defaultdict

trades = []
with open('btcusd/trade_log.csv', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        ts, direction, entry, exit_p, profit_str = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            profit = float(profit_str)
            entry = float(entry)
            exit_p = float(exit_p)
            dt = datetime.fromisoformat(ts.replace('+00:00', ''))
            trades.append({'dt': dt, 'dir': direction, 'entry': entry, 'exit': exit_p, 'profit': profit})
        except:
            continue

print(f"=== BTCUSD Deep Dive ({len(trades)} trades) ===\n")

# 1. BUY vs SELL breakdown
buys = [t for t in trades if t['dir'] == 'BUY']
sells = [t for t in trades if t['dir'] == 'SELL']
print("--- Direction Breakdown ---")
for label, subset in [('BUY', buys), ('SELL', sells)]:
    if not subset:
        print(f"  {label}: 0 trades")
        continue
    w = sum(1 for t in subset if t['profit'] > 0)
    pnl = sum(t['profit'] for t in subset)
    avg_w = sum(t['profit'] for t in subset if t['profit'] > 0) / max(w, 1)
    avg_l = sum(t['profit'] for t in subset if t['profit'] < 0) / max(len(subset) - w, 1)
    print(f"  {label}: {len(subset)} trades | WR: {w/len(subset)*100:.0f}% | P/L: ${pnl:+.2f} | Avg W: ${avg_w:.2f} / Avg L: ${avg_l:.2f}")

# 2. Win/Loss size distribution
print("\n--- Win/Loss Distribution ---")
wins = [t['profit'] for t in trades if t['profit'] > 0]
losses = [t['profit'] for t in trades if t['profit'] < 0]
zeros = [t for t in trades if t['profit'] == 0]
print(f"  Winners: {len(wins)} | Avg: ${sum(wins)/len(wins):.2f} | Max: ${max(wins):.2f} | Min: ${min(wins):.2f}")
print(f"  Losers:  {len(losses)} | Avg: ${sum(losses)/len(losses):.2f} | Max: ${min(losses):.2f} | Min: ${max(losses):.2f}")
print(f"  Breakeven: {len(zeros)}")

# 3. Risk/Reward ratio
avg_win = sum(wins) / len(wins) if wins else 0
avg_loss = abs(sum(losses) / len(losses)) if losses else 1
print(f"\n  Risk/Reward Ratio: {avg_win/avg_loss:.2f} (avg win / avg loss)")
print(f"  Required WR for breakeven: {avg_loss/(avg_win+avg_loss)*100:.0f}%")

# 4. Consecutive losses
print("\n--- Streak Analysis ---")
max_loss_streak = 0
current_streak = 0
streaks = []
for t in trades:
    if t['profit'] < 0:
        current_streak += 1
        max_loss_streak = max(max_loss_streak, current_streak)
    else:
        if current_streak > 0:
            streaks.append(current_streak)
        current_streak = 0
print(f"  Max consecutive losses: {max_loss_streak}")
print(f"  Avg loss streak: {sum(streaks)/len(streaks):.1f}" if streaks else "  No loss streaks")

# 5. Hourly performance
print("\n--- Hourly P/L (UTC) ---")
hourly = defaultdict(list)
for t in trades:
    hourly[t['dt'].hour].append(t['profit'])
for h in sorted(hourly.keys()):
    t_list = hourly[h]
    pnl = sum(t_list)
    wr = sum(1 for x in t_list if x > 0) / len(t_list) * 100
    print(f"  {h:02d}:00 | {len(t_list):3d} trades | WR: {wr:.0f}% | P/L: ${pnl:+.2f}")

# 6. Daily P/L trend
print("\n--- Daily P/L (last 10 days) ---")
daily = defaultdict(list)
for t in trades:
    daily[str(t['dt'].date())].append(t['profit'])
for d in sorted(daily.keys())[-10:]:
    t_list = daily[d]
    pnl = sum(t_list)
    w = sum(1 for x in t_list if x > 0)
    bar = '+' * int(abs(pnl) / 5) if pnl > 0 else '-' * int(abs(pnl) / 5)
    print(f"  {d} | {len(t_list):3d}t | WR:{w/len(t_list)*100:3.0f}% | ${pnl:+7.2f} | {bar}")

# 7. Config check
import json
try:
    with open('btcusd/config.json', 'r') as f:
        cfg = json.load(f)
    print("\n--- Current Config ---")
    print(f"  SL: {cfg.get('stop_loss_pips', 'N/A')} pips | TP: {cfg.get('take_profit_pips', 'N/A')} pips")
    print(f"  Confidence threshold: {cfg.get('confidence_threshold', 'N/A')}")
    print(f"  Max lot: {cfg.get('max_lot_size', 'N/A')} | Risk: {cfg.get('risk_percent', 'N/A')}%")
    print(f"  Cooldown: {cfg.get('trade_cooldown_minutes', 'N/A')} min")
    sm = cfg.get('smart_filters', {})
    print(f"  Smart filters: vol={sm.get('volatility_filter', 'N/A')}, crash={sm.get('crash_detector', 'N/A')}, adaptive_cooldown={sm.get('adaptive_cooldown', 'N/A')}")
except:
    pass

try:
    with open('btcusd/ml_config.json', 'r') as f:
        ml = json.load(f)
    print(f"\n--- ML Config ---")
    print(f"  Model: {ml.get('model_type', 'N/A')} | Timeframe: {ml.get('timeframe', 'N/A')}")
    print(f"  Confidence threshold: {ml.get('confidence_threshold', 'N/A')}")
    print(f"  TP: {ml.get('tp_threshold', 'N/A')}% | SL: {ml.get('sl_threshold', 'N/A')}%")
except:
    pass
