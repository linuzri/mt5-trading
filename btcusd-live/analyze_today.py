import csv
from datetime import datetime
from collections import defaultdict

trades = []
with open('trade_log.csv', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if '2026-02-19' in parts[0]:
            ts = parts[0]
            direction = parts[1]
            entry = float(parts[2])
            exit_p = float(parts[3])
            pnl = float(parts[4])
            hour = int(ts[11:13])
            trades.append({'ts': ts, 'dir': direction, 'entry': entry, 'exit': exit_p, 'pnl': pnl, 'hour': hour})

wins = [t for t in trades if t['pnl'] > 0]
losses = [t for t in trades if t['pnl'] <= 0]
total_pnl = sum(t['pnl'] for t in trades)
best = max(trades, key=lambda t: t['pnl'])
worst = min(trades, key=lambda t: t['pnl'])
avg_win = sum(t['pnl'] for t in wins)/len(wins) if wins else 0
avg_loss = sum(t['pnl'] for t in losses)/len(losses) if losses else 0
win_sum = sum(t['pnl'] for t in wins)
loss_sum = sum(t['pnl'] for t in losses)

print(f"Total trades: {len(trades)}")
print(f"Wins: {len(wins)} | Losses: {len(losses)}")
print(f"Win rate: {len(wins)/len(trades)*100:.1f}%")
print(f"Total P/L: ${total_pnl:.2f}")
print(f"Gross profit: ${win_sum:.2f} | Gross loss: ${loss_sum:.2f}")
print(f"Avg win: ${avg_win:.4f} | Avg loss: ${avg_loss:.4f}")
print(f"Best: ${best['pnl']:.4f} at {best['ts'][:19]}")
print(f"Worst: ${worst['pnl']:.4f} at {worst['ts'][:19]}")
if loss_sum != 0:
    print(f"Profit factor: {abs(win_sum/loss_sum):.2f}")

# Hourly
hourly = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
for t in trades:
    h = t['hour']
    hourly[h]['count'] += 1
    hourly[h]['pnl'] += t['pnl']
    if t['pnl'] > 0: hourly[h]['wins'] += 1

print("\n--- Hourly Breakdown (UTC) ---")
for h in sorted(hourly.keys()):
    d = hourly[h]
    wr = d['wins']/d['count']*100 if d['count'] else 0
    myt = h + 8
    print(f"  {h:02d}:00 UTC ({myt:02d}:00 MYT) | {d['count']:2d} trades | P/L: ${d['pnl']:+.2f} | WR: {wr:.0f}%")

# Gaps
print("\n--- Trading Gaps (>30min) ---")
for i in range(1, len(trades)):
    t1 = datetime.fromisoformat(trades[i-1]['ts'][:26])
    t2 = datetime.fromisoformat(trades[i]['ts'][:26])
    gap = (t2-t1).total_seconds()/60
    if gap > 30:
        print(f"  {trades[i-1]['ts'][11:16]} -> {trades[i]['ts'][11:16]} UTC = {gap:.0f} min gap")

prices = [t['entry'] for t in trades] + [t['exit'] for t in trades]
print(f"\nPrice range: ${min(prices):.2f} - ${max(prices):.2f} (spread: ${max(prices)-min(prices):.2f})")

# Streaks
max_win_streak = max_loss_streak = cur_win = cur_loss = 0
for t in trades:
    if t['pnl'] > 0:
        cur_win += 1; cur_loss = 0
    else:
        cur_loss += 1; cur_win = 0
    max_win_streak = max(max_win_streak, cur_win)
    max_loss_streak = max(max_loss_streak, cur_loss)
print(f"Max win streak: {max_win_streak} | Max loss streak: {max_loss_streak}")

# Drawdown
running = 0; peak = 0; max_dd = 0
for t in trades:
    running += t['pnl']
    peak = max(peak, running)
    max_dd = max(max_dd, peak - running)
print(f"Max drawdown: ${max_dd:.2f}")
print(f"All SELL: {all(t['dir']=='SELL' for t in trades)}")
print(f"First trade: {trades[0]['ts'][:19]} | Last trade: {trades[-1]['ts'][:19]}")
