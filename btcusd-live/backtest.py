import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import json
from itertools import product
import joblib
import os

# ML imports - check if model files exist
ML_AVAILABLE = os.path.exists("models/random_forest_btcusd.pkl") and os.path.exists("models/scaler_btcusd.pkl")

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
# For 6 months of M5 bars: 6 months ~ 180 days x 24 hours x 12 bars/hour = 51,840 bars
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

# Spread and slippage modeling — critical for realistic backtesting
# Estimated spread cost per trade (round-trip: entry + exit)
# BTCUSD typical spread: $20-50 on Pepperstone M5; EURUSD: ~1 pip
spread_config = config.get("backtest_spread", {})
SPREAD_PIPS = spread_config.get("spread_pips", 0.0003 if symbol.endswith("USD") and symbol.startswith("BTC") else 0.00015)
SLIPPAGE_PIPS = spread_config.get("slippage_pips", 0.0001)  # Additional slippage per trade
COMMISSION_PER_LOT = spread_config.get("commission_per_lot", 0)  # Some brokers charge per-lot commission
TOTAL_COST_PCT = SPREAD_PIPS + SLIPPAGE_PIPS  # Total cost as fraction of price per round-trip trade
print(f"[BACKTEST] Spread model: spread={SPREAD_PIPS:.5f}, slippage={SLIPPAGE_PIPS:.5f}, total_cost={TOTAL_COST_PCT:.5f} ({TOTAL_COST_PCT*100:.3f}%)")


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
        # Simulate P/L with spread/slippage costs
        df['returns'] = df['close'].pct_change().shift(-1)  # next bar return
        df['strategy_gross'] = df['signal'].shift(1) * df['returns']
        # Identify trades (when signal changes) — spread cost applies on each trade entry
        df['trade'] = df['signal'].diff().fillna(0).abs() > 0
        # Deduct spread + slippage cost on every trade
        df['trade_cost'] = df['trade'].astype(float) * TOTAL_COST_PCT
        df['strategy'] = df['strategy_gross'] - df['trade_cost']
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
        total_return_gross = df['strategy_gross'].sum()
        total_return = df['strategy'].sum()
        total_costs = df['trade_cost'].sum()
        total_return_pips = total_return / pip_size
        total_return_currency = total_return * (df['close'].iloc[-1] / pip_size) * tick_value * lot
        total_costs_pips = total_costs / pip_size
        num_trades = df['trade'].sum()
        print(f"\nBacktest Results (with spread/slippage model):")
        print(f"  Gross return:  {total_return_gross/pip_size:.2f} pips")
        print(f"  Total costs:   -{total_costs_pips:.2f} pips ({num_trades:.0f} trades × {TOTAL_COST_PCT*100:.3f}%)")
        print(f"  Net return:    {total_return_pips:.2f} pips | {total_return_currency:.2f} {account_info.currency}")
        print(df[['time', 'close', 'ma50', 'ma200', 'signal', 'strategy']].tail(10))
    else:
        print("Failed to get historical data for backtest.")

def run_backtest_optimized():
    # Define ranges for moving average periods (fast for M1 scalping)
    short_ma_range = range(5, 31, 5)    # 5, 10, 15, 20, 25, 30
    long_ma_range = range(20, 81, 10)   # 20, 30, 40, 50, 60, 70, 80
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

# --- ML Backtest Function ---
def run_ml_backtest(df, ml_config_path="ml_config.json"):
    """
    Run backtest using the trained ML model.
    Returns total return in price units.
    """
    if not ML_AVAILABLE:
        print("[ML Backtest] ML module not available")
        return None, {}

    # Load ML config
    try:
        with open(ml_config_path, "r") as f:
            ml_config = json.load(f)
    except FileNotFoundError:
        print("[ML Backtest] ml_config.json not found")
        return None, {}

    # Load trained model and scaler
    model_path = ml_config.get("paths", {}).get("model_file", "models/random_forest_btcusd.pkl")
    scaler_path = ml_config.get("paths", {}).get("scaler_file", "models/scaler_btcusd.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[ML Backtest] Model or scaler not found. Train the model first.")
        return None, {}

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"[ML Backtest] Failed to load model: {e}")
        return None, {}

    # Get prediction thresholds (matching live trading logic)
    confidence_threshold = ml_config.get("prediction", {}).get("confidence_threshold", 0.40)
    min_prob_diff = ml_config.get("prediction", {}).get("min_probability_diff", 0.10)
    max_hold_probability = ml_config.get("prediction", {}).get("max_hold_probability", 0.50)

    # Engineer features for the entire dataset
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={'time': 'time', 'open': 'open', 'high': 'high',
                                       'low': 'low', 'close': 'close', 'tick_volume': 'tick_volume'})

    # Calculate features manually (same as feature_engineering.py)
    closes = df_copy['close'].values
    highs = df_copy['high'].values
    lows = df_copy['low'].values
    volumes = df_copy['tick_volume'].values if 'tick_volume' in df_copy.columns else df_copy['real_volume'].values

    # RSI
    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    exp1 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    # ATR
    high_low = pd.Series(highs) - pd.Series(lows)
    high_close = abs(pd.Series(highs) - pd.Series(closes).shift())
    low_close = abs(pd.Series(lows) - pd.Series(closes).shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    # Bollinger Bands
    bb_ma = pd.Series(closes).rolling(window=20).mean()
    bb_std = pd.Series(closes).rolling(window=20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma

    # Volume ratio
    vol_ma = pd.Series(volumes).rolling(window=20).mean()
    volume_ratio = pd.Series(volumes) / vol_ma

    # Price changes
    price_change_1 = pd.Series(closes).pct_change(1)
    price_change_5 = pd.Series(closes).pct_change(5)

    # Create features dataframe
    features_df = pd.DataFrame({
        'rsi_14': rsi,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'atr_14': atr,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_width': bb_width,
        'volume_ratio': volume_ratio,
        'price_change_1min': price_change_1,
        'price_change_5min': price_change_5
    })

    # Generate predictions
    signals = np.zeros(len(df_copy))
    confidences = np.zeros(len(df_copy))

    # Start from index 50 to have enough data for features
    for i in range(50, len(df_copy)):
        try:
            # Get features for this row
            row_features = features_df.iloc[i:i+1].values

            if np.isnan(row_features).any():
                continue

            # Scale features
            row_scaled = scaler.transform(row_features)

            # Get prediction and probabilities
            # Model classes: 0=sell, 1=buy, 2=hold
            probs = model.predict_proba(row_scaled)[0]
            sell_prob = probs[0]
            buy_prob = probs[1]
            hold_prob = probs[2]

            # CRITICAL: Respect HOLD signal when probability is high (matching live logic)
            if hold_prob > max_hold_probability:
                continue  # Skip this bar - HOLD signal too strong

            # Determine signal based on BUY vs SELL only
            if buy_prob > sell_prob:
                signal_type = 'buy'
                signal_confidence = buy_prob
            else:
                signal_type = 'sell'
                signal_confidence = sell_prob

            # Apply confidence threshold on the chosen signal
            if signal_confidence < confidence_threshold:
                continue

            # Check probability difference between BUY and SELL
            prob_diff = abs(buy_prob - sell_prob)
            if prob_diff < min_prob_diff:
                continue

            # Valid signal
            if signal_type == 'buy':
                signals[i] = 1
            else:
                signals[i] = -1
            confidences[i] = signal_confidence

        except Exception as e:
            continue

    # Calculate returns
    returns = pd.Series(closes).pct_change().shift(-1)
    strategy_returns = pd.Series(signals).shift(1) * returns
    total_return = strategy_returns.sum()

    # Calculate stats
    trades = np.sum(np.abs(np.diff(signals)) > 0)
    avg_confidence = np.mean(confidences[confidences > 0]) if np.any(confidences > 0) else 0

    ml_params = {
        'confidence_threshold': confidence_threshold,
        'min_probability_diff': min_prob_diff,
        'max_hold_probability': max_hold_probability,
        'total_trades': int(trades),
        'avg_confidence': float(avg_confidence),
        'total_return': float(total_return)
    }

    return total_return, ml_params

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
    # Fast MA ranges for M1 scalping
    print("\n[MA Crossover Optimization]")
    short_ma_range = range(5, 31, 5)    # 5, 10, 15, 20, 25, 30
    long_ma_range = range(20, 81, 10)   # 20, 30, 40, 50, 60, 70, 80
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

    # --- ML Random Forest ---
    print("\n[ML Random Forest Backtest]")
    if ML_AVAILABLE:
        ml_return, ml_params = run_ml_backtest(df)
        if ml_return is not None:
            results['ml_random_forest'] = ml_params
            print(f"ML Random Forest: return={ml_return/pip_size:.2f} pips, trades={ml_params.get('total_trades', 0)}, avg_conf={ml_params.get('avg_confidence', 0)*100:.1f}%")
        else:
            print("ML backtest failed - model may need training")
    else:
        print("ML module not available")

    # Save all best params/results
    with open("strategy_params.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll best parameters/results saved to strategy_params.json")

    # --- STRATEGY RANKING (no auto-change) ---
    print("\n" + "="*50)
    print("STRATEGY PERFORMANCE RANKING")
    print("="*50)

    # Get current strategy from config
    with open("config.json", "r") as f:
        current_config = json.load(f)
    current_strategy = current_config.get("strategy", "unknown")

    # Sort strategies by return
    sorted_strategies = sorted(results.items(), key=lambda x: x[1].get('total_return', 0), reverse=True)

    best_strategy = None
    best_return = -float('inf')

    for rank, (strat_name, strat_data) in enumerate(sorted_strategies, 1):
        ret = strat_data.get('total_return', 0)
        ret_pips = ret / pip_size
        status = ""
        if rank == 1 and ret > 0:
            best_strategy = strat_name
            best_return = ret
            status = " <-- BEST"
        if strat_name == current_strategy:
            status += " (CURRENT)"
        if ret <= 0:
            status += " (negative/zero)"
        print(f"  {rank}. {strat_name}: {ret_pips:.2f} pips{status}")

    print("="*50)
    print(f"Current strategy: {current_strategy}")
    if best_strategy:
        print(f"Best performing: {best_strategy} ({best_return/pip_size:.2f} pips)")
    print("="*50)
    print("NOTE: config.json NOT changed. Manually set strategy if needed.")
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

    # ML Random Forest
    ml = strategy_params.get('ml_random_forest', {})
    if ml and ML_AVAILABLE:
        ml_return, ml_fwd_params = run_ml_backtest(df)
        if ml_return is not None:
            print(f"ML Random Forest Forward: return={ml_return/pip_size:.2f} pips, trades={ml_fwd_params.get('total_trades', 0)}")
        else:
            print("ML Forward test failed")
    elif not ML_AVAILABLE:
        print("ML Forward: module not available")

if __name__ == "__main__":
    print(f"--- Running Backtest Optimization ({backtest_period_days} days) ---")
    run_all_strategies_backtest()
    # Load best params for forward test
    with open("strategy_params.json", "r") as f:
        strategy_params = json.load(f)
    print(f"\n--- Running Forward Test ({forward_test_period_days} days) ---")
    run_all_strategies_forward_test(strategy_params)
    mt5.shutdown()
