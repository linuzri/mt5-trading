import csv
from datetime import datetime
from collections import defaultdict

bots = {'btcusd': 'btcusd/trade_log.csv', 'xauusd': 'xauusd/trade_log.csv', 'eurusd': 'eurusd/trade_log.csv'}
days = defaultdict(lambda: defaultdict(list))
all_trades = defaultdict(list)

for bot, path in bots.items():
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5:
                    continue
                ts, direction, entry, exit_p, profit_str = parts[0], parts[1], parts[2], parts[3], parts[4]
                try:
                    profit = float(profit_str)
                except:
                    continue
                try:
                    dt = datetime.fromisoformat(ts.replace('+00:00', ''))
                    d = str(dt.date())
                    days[d][bot].append(profit)
                    all_trades[bot].append(profit)
                except:
                    pass
    except:
        pass

print("=== Last 7 Days ===")
for d in sorted(days.keys())[-7:]:
    parts_str = []
    total = 0
    for bot in ['btcusd', 'xauusd', 'eurusd']:
        t = days[d].get(bot, [])
        pnl = sum(t)
        total += pnl
        w = sum(1 for x in t if x > 0)
        parts_str.append(f"{bot.upper()}: {len(t)}t W{w} ${pnl:+.2f}")
    print(f"  {d} | {' | '.join(parts_str)} | TOTAL: ${total:+.2f}")

print("\n=== Overall Stats (All Time) ===")
grand_total = 0
for bot in ['btcusd', 'xauusd', 'eurusd']:
    t = all_trades[bot]
    if not t:
        print(f"  {bot.upper()}: No trades")
        continue
    wins = sum(1 for x in t if x > 0)
    losses = sum(1 for x in t if x < 0)
    total_pnl = sum(t)
    grand_total += total_pnl
    avg_win = sum(x for x in t if x > 0) / max(wins, 1)
    avg_loss = sum(x for x in t if x < 0) / max(losses, 1)
    gross_loss = abs(sum(x for x in t if x < 0))
    pf = sum(x for x in t if x > 0) / gross_loss if gross_loss > 0 else 999
    wr = wins / len(t) * 100
    print(f"  {bot.upper()}: {len(t)} trades | WR: {wr:.0f}% | P/L: ${total_pnl:+.2f} | PF: {pf:.2f} | Avg W: ${avg_win:.2f} / Avg L: ${avg_loss:.2f}")

print(f"\n  GRAND TOTAL P/L: ${grand_total:+.2f}")
