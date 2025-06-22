import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import json
from itertools import product

# Read credentials from config.json
with open("config.json", "r") as f:
    config = json.load(f)
login = config["login"]
password = config["password"]
server = config["server"]

# Connect to MT5 (default path or existing instance)
if not mt5.initialize(login=login, password=password, server=server):
    print("MT5 initialize() failed, error code=", mt5.last_error())
    mt5.shutdown()
    quit()
print("Connected to MT5:", mt5.terminal_info().name)

account_info = mt5.account_info()
if account_info:
    print(f"Account balance: {account_info.balance:.2f} {account_info.currency}")
else:
    print("Failed to get account info.")

symbol = "EURUSD"  # You can change this to any symbol available in your account
# For 3 months of M5 bars: 3 months ≈ 90 days × 24 hours × 12 bars/hour = 25,920 bars
num_bars = 25920  # Number of bars for backtest (approx. 3 months)

# Helper to get pip size for symbol
pip_size = 0.0001 if symbol.endswith('JPY') is False else 0.01
lot = 0.01  # Example lot size for P/L calculation

# Get symbol info for tick value
symbol_info = mt5.symbol_info(symbol)
tick_value = symbol_info.trade_tick_value if symbol_info else 1.0


def run_backtest():
    # Fetch historical M5 data
    start_date = datetime.now(UTC) - timedelta(minutes=num_bars*5)
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, start_date, num_bars)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['close'] = df['close'].astype(float)
        # Compute moving averages
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        df['signal'] = 0
        df.loc[df['ma50'] > df['ma200'], 'signal'] = 1
        df.loc[df['ma50'] < df['ma200'], 'signal'] = -1
        # Simulate P/L (very basic example)
        df['returns'] = df['close'].pct_change().shift(-1)  # next bar return
        df['strategy'] = df['signal'].shift(1) * df['returns']
        # Identify trades (when signal changes)
        df['trade'] = df['signal'].diff().fillna(0).abs() > 0
        trades = df[df['trade'] & (df['signal'].shift(1) != 0)]
        print("Individual trade P/L results:")
        for idx in trades.index:
            trade_time = df.loc[idx, 'time']
            trade_signal = df.loc[idx, 'signal']
            trade_pl_raw = df.loc[idx, 'strategy']
            # Convert to pips
            trade_pl_pips = trade_pl_raw / pip_size if not np.isnan(trade_pl_raw) else float('nan')
            # Convert to account currency (approximate)
            trade_pl_currency = trade_pl_raw * (df.loc[idx, 'close'] / pip_size) * tick_value * lot if not np.isnan(trade_pl_raw) else float('nan')
            print(f"{trade_time} | {'BUY' if trade_signal==1 else 'SELL'} | P/L: {trade_pl_pips:.2f} pips | {trade_pl_currency:.2f} {account_info.currency}")
        total_return = df['strategy'].sum()
        total_return_pips = total_return / pip_size
        total_return_currency = total_return * (df['close'].iloc[-1] / pip_size) * tick_value * lot
        print(f"\nBacktest total return: {total_return_pips:.2f} pips | {total_return_currency:.2f} {account_info.currency}")
        print(df[['time', 'close', 'ma50', 'ma200', 'signal', 'strategy']].tail(10))
    else:
        print("Failed to get historical data for backtest.")

def run_backtest_optimized():
    # Define ranges for moving average periods
    short_ma_range = range(10, 101, 10)   # 10, 20, ..., 100
    long_ma_range = range(100, 301, 20)   # 100, 120, ..., 300
    best_result = None
    best_params = None
    print("\nOptimizing moving average periods...")
    start_date = datetime.now(UTC) - timedelta(minutes=num_bars*5)
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, start_date, num_bars)
    if rates is None or len(rates) == 0:
        print("Failed to get historical data for backtest.")
        return
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['close'] = df['close'].astype(float)
    results = []
    for short_ma, long_ma in product(short_ma_range, long_ma_range):
        if short_ma >= long_ma:
            continue
        df['ma_short'] = df['close'].rolling(window=short_ma).mean()
        df['ma_long'] = df['close'].rolling(window=long_ma).mean()
        df['signal'] = 0
        df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
        df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1
        df['returns'] = df['close'].pct_change().shift(-1)
        df['strategy'] = df['signal'].shift(1) * df['returns']
        returns = df['strategy'].dropna()
        if len(returns) > 1:
            sharpe = returns.mean() / returns.std() * np.sqrt(252*24*12)
            cumulative = (1 + returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
        else:
            sharpe = float('nan')
            max_drawdown = float('nan')
        total_return = df['strategy'].sum()
        results.append((short_ma, long_ma, total_return, sharpe, max_drawdown))
        if best_result is None or total_return > best_result:
            best_result = total_return
            best_params = (short_ma, long_ma)
    print("\nTop 5 parameter sets by total return:")
    results.sort(key=lambda x: x[2], reverse=True)
    for i, (short_ma, long_ma, total_return, sharpe, max_drawdown) in enumerate(results[:5]):
        total_return_pips = total_return / pip_size
        total_return_currency = total_return * (df['close'].iloc[-1] / pip_size) * tick_value * lot
        print(f"{i+1}. Short MA: {short_ma}, Long MA: {long_ma} | Return: {total_return_pips:.2f} pips | {total_return_currency:.2f} {account_info.currency} | Sharpe: {sharpe:.2f} | Max DD: {max_drawdown:.2%}")
    if best_params:
        print(f"\nBest parameters: Short MA = {best_params[0]}, Long MA = {best_params[1]}")
        print(f"Best total return: {best_result / pip_size:.2f} pips | {(best_result * (df['close'].iloc[-1] / pip_size) * tick_value * lot):.2f} {account_info.currency}")
        # Save best params to file
        with open("strategy_params.json", "w") as f:
            json.dump({"short_ma": best_params[0], "long_ma": best_params[1]}, f)
        print("Best parameters saved to strategy_params.json")
    else:
        print("No profitable parameter set found.")

def run_forward_test():
    # Fetch the latest 250 1-minute bars
    utc_now = datetime.now(UTC)
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_now - timedelta(minutes=250), 250)
    if rates is not None and len(rates) > 0:
        closes = np.array([bar[4] for bar in rates])
        short_ma = np.mean(closes[-50:])
        long_ma = np.mean(closes[-200:])
        print(f"Forward Test: short_ma={short_ma:.5f}, long_ma={long_ma:.5f}")
        if short_ma > long_ma:
            print("Signal: BUY")
        elif short_ma < long_ma:
            print("Signal: SELL")
        else:
            print("Signal: HOLD")
    else:
        print("Failed to get bars for forward test.")

if __name__ == "__main__":
    print("--- Running Backtest Optimization (3 months) ---")
    run_backtest_optimized()
    print("\n--- Running Forward Test (Demo) ---")
    run_forward_test()
    mt5.shutdown()
