import csv
bots = {'BTCUSD': r'C:\Users\Nazri Hussain\projects\mt5-trading\btcusd\trade_log.csv',
        'XAUUSD': r'C:\Users\Nazri Hussain\projects\mt5-trading\xauusd\trade_log.csv',
        'EURUSD': r'C:\Users\Nazri Hussain\projects\mt5-trading\eurusd\trade_log.csv'}
total_pnl = 0
total_trades = 0
for bot, path in bots.items():
    trades, wins, pnl = 0, 0, 0
    try:
        for row in csv.reader(open(path)):
            if len(row) < 5: continue
            if '2026-02-11' in row[0]:
                trades += 1
                p = float(row[4]) if row[4] != 'N/A' else 0
                pnl += p
                if p > 0: wins += 1
    except: pass
    wr = (wins/trades*100) if trades > 0 else 0
    print(f"{bot}: {trades} trades, {wins}W/{trades-wins}L, WR: {wr:.0f}%, P/L: ${pnl:.2f}")
    total_pnl += pnl
    total_trades += trades
print(f"\nTOTAL: {total_trades} trades, P/L: ${total_pnl:.2f}")
