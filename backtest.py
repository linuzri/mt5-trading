import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import json
from itertools import product

# Read credentials from mt5_auth.json
with open("mt5_auth.json", "r") as f:
    auth = json.load(f)
login = auth["login"]
password = auth["password"]
server = auth["server"]
# Read other config from config.json
with open("config.json", "r") as f:
    config = json.load(f)
backtest_period_days = config.get("backtest_period_days", 365)
forward_test_period_days = config.get("forward_test_period_days", 28)
timeframe_str = config.get("timeframe", "M5")
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}
timeframe = TIMEFRAME_MAP.get(timeframe_str.upper(), mt5.TIMEFRAME_M5)

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

# Symbol is now configurable via config.json
symbol = config.get("symbol", "BTCUSD")
# For 6 months of M5 bars: 6 months ≈ 180 days × 24 hours × 12 bars/hour = 51,840 bars
num_bars = 51840  # Number of bars for backtest (approx. 6 months)

# Helper to get pip size for symbol (crypto uses point value from broker)
symbol_info = mt5.symbol_info(symbol)
if symbol_info:
    pip_size = symbol_info.point  # Use broker's point value (0.01 for BTCUSD, 0.0001 for EURUSD, etc.)
else:
    pip_size = 0.0001 if not symbol.endswith('JPY') else 0.01
lot = 0.01  # Example lot size for P/L calculation

# Get symbol info for tick value
symbol_info = mt5.symbol_info(symbol)
tick_value = symbol_info.trade_tick_value if symbol_info else 1.0


def run_backtest():
    # Fetch historical M5 data using copy_rates_range for better reliability
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=backtest_period_days)
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
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
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=backtest_period_days)
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
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

# --- Indicator helpers ---
def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    up = delta.clip(min=0)
    down = -1 * delta.clip(max=0)
    ma_up = pd.Series(up).rolling(window=period, min_periods=period).mean()
    ma_down = pd.Series(down).rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, fast=12, slow=26, signal=9):
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(prices, period=20, stddev=2):
    series = pd.Series(prices)
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + stddev * std
    lower = ma - stddev * std
    return ma, upper, lower

# --- Multi-strategy backtest ---
def run_all_strategies_backtest():
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=backtest_period_days)
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Failed to get historical data for backtest. Error: {mt5.last_error()}")
        print(f"Trying with shorter period...")
        # Try with 180 days if full period fails
        start_date = end_date - timedelta(days=180)
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            print("Still no data available. Exiting backtest.")
            return
    print(f"Loaded {len(rates)} bars for backtesting ({len(rates) / (24*12):.1f} days of data)")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['close'] = df['close'].astype(float)
    closes = df['close'].values
    results = {}

    # --- MA Crossover ---
    print("\n[MA Crossover Optimization]")
    short_ma_range = range(10, 101, 10)
    long_ma_range = range(100, 301, 20)
    best_ma = None
    best_ma_result = -np.inf
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
        total_return = df['strategy'].sum()
        if total_return > best_ma_result:
            best_ma_result = total_return
            best_ma = (short_ma, long_ma)
    results['ma_crossover'] = {'short_ma': best_ma[0], 'long_ma': best_ma[1], 'total_return': best_ma_result}
    print(f"Best MA: short={best_ma[0]}, long={best_ma[1]}, return={best_ma_result/pip_size:.2f} pips")

    # --- RSI ---
    print("\n[RSI Optimization]")
    rsi_range = range(7, 31, 1)
    best_rsi = None
    best_rsi_result = -np.inf
    for rsi_period in rsi_range:
        rsi = compute_rsi(closes, rsi_period)
        signal = np.zeros_like(closes)
        if len(rsi) > 0:
            signal[-len(rsi):][rsi < 30] = 1
            signal[-len(rsi):][rsi > 70] = -1
        returns = pd.Series(signal).shift(1) * pd.Series(closes).pct_change().shift(-1)
        total_return = returns.sum()
        if total_return > best_rsi_result:
            best_rsi_result = total_return
            best_rsi = rsi_period
    results['rsi'] = {'rsi_period': best_rsi, 'total_return': best_rsi_result}
    print(f"Best RSI period: {best_rsi}, return={best_rsi_result/pip_size:.2f} pips")

    # --- MACD ---
    print("\n[MACD Optimization]")
    fast_range = range(8, 17, 2)
    slow_range = range(19, 31, 2)
    signal_range = range(7, 13, 2)
    best_macd = None
    best_macd_result = -np.inf
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            for sig in signal_range:
                macd, sig_line = compute_macd(closes, fast, slow, sig)
                signal = np.zeros_like(closes)
                if len(macd) > 1 and len(sig_line) > 1:
                    cross_up = (macd.shift(1) < sig_line.shift(1)) & (macd > sig_line)
                    cross_down = (macd.shift(1) > sig_line.shift(1)) & (macd < sig_line)
                    signal[cross_up.index[cross_up]] = 1
                    signal[cross_down.index[cross_down]] = -1
                returns = pd.Series(signal).shift(1) * pd.Series(closes).pct_change().shift(-1)
                total_return = returns.sum()
                if total_return > best_macd_result:
                    best_macd_result = total_return
                    best_macd = (fast, slow, sig)
    results['macd'] = {'macd_fast': best_macd[0], 'macd_slow': best_macd[1], 'macd_signal': best_macd[2], 'total_return': best_macd_result}
    print(f"Best MACD: fast={best_macd[0]}, slow={best_macd[1]}, signal={best_macd[2]}, return={best_macd_result/pip_size:.2f} pips")

    # --- Bollinger Bands ---
    print("\n[Bollinger Bands Optimization]")
    boll_period_range = range(10, 31, 2)
    boll_stddev_range = [1.5, 2, 2.5, 3]
    best_boll = None
    best_boll_result = -np.inf
    for period in boll_period_range:
        for stddev in boll_stddev_range:
            ma, upper, lower = compute_bollinger_bands(closes, period, stddev)
            signal = np.zeros_like(closes)
            if len(upper) > 0 and len(lower) > 0:
                signal[(closes > upper)] = 1
                signal[(closes < lower)] = -1
            returns = pd.Series(signal).shift(1) * pd.Series(closes).pct_change().shift(-1)
            total_return = returns.sum()
            if total_return > best_boll_result:
                best_boll_result = total_return
                best_boll = (period, stddev)
    results['bollinger'] = {'bollinger_period': best_boll[0], 'bollinger_stddev': best_boll[1], 'total_return': best_boll_result}
    print(f"Best Bollinger: period={best_boll[0]}, stddev={best_boll[1]}, return={best_boll_result/pip_size:.2f} pips")

    # Save all best params/results
    with open("strategy_params.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll best parameters/results saved to strategy_params.json")

    # --- AUTO-SELECT BEST STRATEGY ---
    # Find the strategy with the highest total_return
    best_strategy = None
    best_return = -float('inf')

    print("\n" + "="*50)
    print("STRATEGY PERFORMANCE RANKING")
    print("="*50)

    # Sort strategies by return
    sorted_strategies = sorted(results.items(), key=lambda x: x[1].get('total_return', 0), reverse=True)

    for rank, (strat_name, strat_data) in enumerate(sorted_strategies, 1):
        ret = strat_data.get('total_return', 0)
        ret_pips = ret / pip_size
        status = ""
        if rank == 1 and ret > 0:
            best_strategy = strat_name
            best_return = ret
            status = " <-- BEST (SELECTED)"
        elif ret <= 0:
            status = " (negative/zero return)"
        print(f"  {rank}. {strat_name}: {ret_pips:.2f} pips{status}")

    # Update config.json with the best strategy
    if best_strategy and best_return > 0:
        with open("config.json", "r") as f:
            current_config = json.load(f)

        old_strategy = current_config.get("strategy", "unknown")
        current_config["strategy"] = best_strategy

        with open("config.json", "w") as f:
            json.dump(current_config, f, indent=2)

        print("="*50)
        print(f"AUTO-SELECTED: {best_strategy}")
        print(f"Previous strategy: {old_strategy}")
        print(f"Expected return: {best_return/pip_size:.2f} pips")
        print("config.json has been updated!")
        print("="*50)
    else:
        print("="*50)
        print("WARNING: No profitable strategy found!")
        print("Keeping current strategy in config.json")
        print("="*50)

def run_all_strategies_forward_test(strategy_params):
    print("\n--- Forward Test Results (Most Recent Data) ---")
    # Use last N days of M5 bars for forward test
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=forward_test_period_days)
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print("Failed to get bars for forward test.")
        return
    df = pd.DataFrame(rates)
    df['close'] = df['close'].astype(float)
    closes = df['close'].values
    # Use broker's point value for pip_size
    sym_info = mt5.symbol_info(symbol)
    pip_size = sym_info.point if sym_info else 0.0001
    # MA Crossover
    ma = strategy_params.get('ma_crossover', {})
    if ma:
        short_ma = ma['short_ma']
        long_ma = ma['long_ma']
        df['ma_short'] = df['close'].rolling(window=short_ma).mean()
        df['ma_long'] = df['close'].rolling(window=long_ma).mean()
        df['signal'] = 0
        df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
        df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1
        df['returns'] = df['close'].pct_change().shift(-1)
        df['strategy'] = df['signal'].shift(1) * df['returns']
        total_return = df['strategy'].sum()
        print(f"MA Crossover Forward: short={short_ma}, long={long_ma}, return={total_return/pip_size:.2f} pips")
    # RSI
    rsi = strategy_params.get('rsi', {})
    if rsi:
        rsi_period = rsi['rsi_period']
        rsi_vals = compute_rsi(closes, rsi_period)
        signal = np.zeros_like(closes)
        if len(rsi_vals) > 0:
            signal[-len(rsi_vals):][rsi_vals < 30] = 1
            signal[-len(rsi_vals):][rsi_vals > 70] = -1
        returns = pd.Series(signal).shift(1) * pd.Series(closes).pct_change().shift(-1)
        total_return = returns.sum()
        print(f"RSI Forward: period={rsi_period}, return={total_return/pip_size:.2f} pips")
    # MACD
    macd = strategy_params.get('macd', {})
    if macd:
        fast = macd['macd_fast']
        slow = macd['macd_slow']
        sig = macd['macd_signal']
        macd_vals, sig_line = compute_macd(closes, fast, slow, sig)
        signal = np.zeros_like(closes)
        if len(macd_vals) > 1 and len(sig_line) > 1:
            cross_up = (macd_vals.shift(1) < sig_line.shift(1)) & (macd_vals > sig_line)
            cross_down = (macd_vals.shift(1) > sig_line.shift(1)) & (macd_vals < sig_line)
            signal[cross_up.index[cross_up]] = 1
            signal[cross_down.index[cross_down]] = -1
        returns = pd.Series(signal).shift(1) * pd.Series(closes).pct_change().shift(-1)
        total_return = returns.sum()
        print(f"MACD Forward: fast={fast}, slow={slow}, signal={sig}, return={total_return/pip_size:.2f} pips")
    # Bollinger Bands
    boll = strategy_params.get('bollinger', {})
    if boll:
        period = boll['bollinger_period']
        stddev = boll['bollinger_stddev']
        ma, upper, lower = compute_bollinger_bands(closes, period, stddev)
        signal = np.zeros_like(closes)
        if len(upper) > 0 and len(lower) > 0:
            signal[(closes > upper)] = 1
            signal[(closes < lower)] = -1
        returns = pd.Series(signal).shift(1) * pd.Series(closes).pct_change().shift(-1)
        total_return = returns.sum()
        print(f"Bollinger Forward: period={period}, stddev={stddev}, return={total_return/pip_size:.2f} pips")

if __name__ == "__main__":
    print(f"--- Running Backtest Optimization ({backtest_period_days} days) ---")
    run_all_strategies_backtest()
    # Load best params for forward test
    with open("strategy_params.json", "r") as f:
        strategy_params = json.load(f)
    print(f"\n--- Running Forward Test ({forward_test_period_days} days) ---")
    run_all_strategies_forward_test(strategy_params)
    mt5.shutdown()
