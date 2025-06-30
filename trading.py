import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import calendar
import time
import json
import subprocess
import sys
import requests
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from pytz import timezone as ZoneInfo  # fallback for older Python

# Read credentials from mt5_auth.json
with open("mt5_auth.json", "r") as f:
    auth = json.load(f)
login = auth["login"]
password = auth["password"]
server = auth["server"]
# Read other config from config.json
with open("config.json", "r") as f:
    config = json.load(f)
sl_pips = config.get("sl_pips", 0.0010)  # default 10 pips
tp_pips = config.get("tp_pips", 0.0020)  # default 20 pips
atr_period = config.get("atr_period", 14)  # ATR period for trailing stop
atr_multiplier = config.get("atr_multiplier", 1.0)  # Multiplier for ATR trailing stop
enable_trailing_stop = config.get("enable_trailing_stop", True)
strategy = config.get("strategy", "ma_crossover")
timeframe_str = config.get("timeframe", "M5")
# Map string to MT5 timeframe constant
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

# Define symbol and lot before the main loop
symbol = "EURUSD"
lot = 0.01

# Define log_file, telegram_cfg, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID before the main loop
log_file = "trade_notifications.log"
telegram_cfg = auth.get("telegram", {})
TELEGRAM_TOKEN = telegram_cfg.get("api_token")
TELEGRAM_CHAT_ID = telegram_cfg.get("chat_id")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[DEBUG] Telegram token or chat_id missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=data, timeout=5)
    #   print(f"[DEBUG] Telegram response: {resp.text}")
    except Exception as e:
        print(f"[DEBUG] Telegram error: {e}")

def log_notify(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now(UTC).isoformat()} {message}\n")
    send_telegram_message(message)

# --- Helper functions for indicators ---
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

# --- Backtest automation on Monday 8:00 AM US Eastern ---
def is_monday_morning_us():
    try:
        eastern = ZoneInfo("America/New_York")
        now_et = datetime.now(eastern)
    except Exception:
        now_et = datetime.now(UTC) - timedelta(hours=4)
    return now_et.weekday() == 0 and now_et.hour >= 8 and now_et.hour < 12  # 8am-12pm ET

last_backtest_monday = None

# Load optimized parameters for the selected strategy if available
try:
    with open("strategy_params.json", "r") as f:
        params = json.load(f)
    strat_params = params.get(strategy, {})
except Exception:
    strat_params = {}

# Set parameters for each strategy (ensure always defined before main loop)
short_ma_period = strat_params.get("short_ma", config.get("short_ma", 10))
long_ma_period = strat_params.get("long_ma", config.get("long_ma", 300))
rsi_period = strat_params.get("rsi_period", config.get("rsi_period", 14))
macd_fast = strat_params.get("macd_fast", config.get("macd_fast", 12))
macd_slow = strat_params.get("macd_slow", config.get("macd_slow", 26))
macd_signal = strat_params.get("macd_signal", config.get("macd_signal", 9))
bollinger_period = strat_params.get("bollinger_period", config.get("bollinger_period", 20))
bollinger_stddev = strat_params.get("bollinger_stddev", config.get("bollinger_stddev", 2))

# --- Trade Management Loop ---
try:
    while True:
        now = datetime.now(UTC)
        try:
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern)
        except Exception:
            now_et = datetime.now(UTC) - timedelta(hours=4)
        # --- Backtest automation inside loop ---
        try:
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern)
        except Exception:
            now_et = datetime.now(UTC) - timedelta(hours=4)
        monday_date = now_et.date() if now_et.weekday() == 0 else None
        if is_monday_morning_us() and (last_backtest_monday != monday_date):
            print("[AUTOMATION] Monday morning US time detected. Running backtest.py...")
            result = subprocess.run([sys.executable, "backtest.py"], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("[AUTOMATION] backtest.py failed:", result.stderr)
            else:
                print("[AUTOMATION] backtest.py completed. Reloading strategy parameters...")
                try:
                    with open("strategy_params.json", "r") as f:
                        params = json.load(f)
                    strat_params = params.get(strategy, {})
                    short_ma_period = strat_params.get("short_ma", config.get("short_ma", 10))
                    long_ma_period = strat_params.get("long_ma", config.get("long_ma", 300))
                    rsi_period = strat_params.get("rsi_period", config.get("rsi_period", 14))
                    macd_fast = strat_params.get("macd_fast", config.get("macd_fast", 12))
                    macd_slow = strat_params.get("macd_slow", config.get("macd_slow", 26))
                    macd_signal = strat_params.get("macd_signal", config.get("macd_signal", 9))
                    bollinger_period = strat_params.get("bollinger_period", config.get("bollinger_period", 20))
                    bollinger_stddev = strat_params.get("bollinger_stddev", config.get("bollinger_stddev", 2))
                    last_backtest_monday = monday_date
                except Exception as e:
                    print("[AUTOMATION] Failed to reload strategy_params.json:", e)
        # (Re)initialize and ensure the symbol is available
        if not mt5.initialize(login=login, password=password, server=server):
            log_notify(f"[ERROR] MT5 initialize() failed, error code={mt5.last_error()}")
            time.sleep(60)
            continue
        if not mt5.symbol_select(symbol, True):
            log_notify(f"[ERROR] Failed to select symbol {symbol}. Waiting...")
            time.sleep(60)
            continue

        # Check if market is open for trading
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            log_notify(f"[ERROR] symbol_info for {symbol} is None. Waiting...")
            time.sleep(60)
            continue
        market_open = (
            symbol_info.visible and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
        )
        if not market_open:
            log_notify(f"[HEARTBEAT] Market is closed for {symbol}. Bot is running. UTC: {now.isoformat()}, ET: {now_et.isoformat()}")
            time.sleep(60)
            continue

        # Fetch the last 250 1-minute bars for EURUSD
        utc_now = datetime.now(UTC)
        rates = mt5.copy_rates_from(symbol, timeframe, utc_now - timedelta(minutes=250), 250)
        if rates is None or len(rates) == 0:
            log_notify(f"Failed to get bars for {symbol}. Waiting...")
            time.sleep(60)
            continue
        closes = np.array([bar[4] for bar in rates])
        # --- Strategy selection ---
        trade_signal = None
        if strategy == "ma_crossover":
            short_ma = np.mean(closes[-short_ma_period:])
            long_ma = np.mean(closes[-long_ma_period:])
            if short_ma > long_ma:
                trade_signal = "buy"
            elif short_ma < long_ma:
                trade_signal = "sell"
        elif strategy == "rsi":
            rsi = compute_rsi(closes, rsi_period)
            last_rsi = rsi.iloc[-1] if len(rsi) > 0 else None
            if last_rsi is not None:
                if last_rsi < 30:
                    trade_signal = "buy"
                elif last_rsi > 70:
                    trade_signal = "sell"
        elif strategy == "macd":
            macd, signal_line = compute_macd(closes, macd_fast, macd_slow, macd_signal)
            if len(macd) > 1 and len(signal_line) > 1:
                if macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]:
                    trade_signal = "buy"
                elif macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1]:
                    trade_signal = "sell"
        elif strategy == "bollinger":
            ma, upper, lower = compute_bollinger_bands(closes, bollinger_period, bollinger_stddev)
            last_close = closes[-1] if len(closes) > bollinger_period else None
            last_upper = upper.iloc[-1] if len(closes) > bollinger_period else None
            last_lower = lower.iloc[-1] if len(closes) > bollinger_period else None
            if last_close is not None and last_upper is not None and last_lower is not None:
                if last_close > last_upper:
                    trade_signal = "buy"
                elif last_close < last_lower:
                    trade_signal = "sell"
        elif strategy == "custom":
            log_notify("Custom strategy not implemented. Please add your logic.")
            trade_signal = None
        else:
            log_notify(f"Unknown strategy: {strategy}. No trades will be made.")
            trade_signal = None
        # log_notify(debug_msg)  # Commented out for future troubleshooting
        # log_notify(f"[DEBUG] trade_signal={trade_signal}")  # Commented out for future troubleshooting

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log_notify(f"Failed to get current tick for {symbol}. Waiting...")
            time.sleep(60)
            continue
        ask = tick.ask
        bid = tick.bid
        if ask == 0 or bid == 0:
            log_notify("Invalid ask/bid price. Waiting...")
            time.sleep(60)
            continue

        # Check for open positions
        positions = mt5.positions_get(symbol=symbol)
        position_type = None
        ticket = None
        entry_price = None
        if positions:
            pos = positions[0]
            position_type = pos.type  # 0=buy, 1=sell
            ticket = pos.ticket
            entry_price = pos.price_open

        # Trading logic with management and notifications
        if trade_signal == "buy":
            if position_type is None:
                # No open position, open BUY
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": ask,
                    "sl": ask - sl_pips,
                    "tp": ask + tp_pips,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "python scalping buy",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] BUY order placed, ticket: {result.order}, price: {ask}")
                    # Log account balance after successful BUY
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after BUY: {balance}")
                else:
                    log_notify(f"BUY order failed, retcode = {result.retcode}")
            elif position_type == 1:
                # Open SELL, but signal is BUY: close SELL, open BUY
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_BUY,
                    "position": ticket,
                    "price": ask,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "close sell, open buy",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                close_result = mt5.order_send(close_request)
                if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] Closed SELL, opened BUY, ticket: {close_result.order}, price: {ask}")
                    # Fetch last deal for profit/loss
                    deals = mt5.history_deals_get(datetime.now(UTC) - timedelta(days=1), datetime.now(UTC))
                    if deals:
                        last_deal = sorted(deals, key=lambda d: d.time, reverse=True)[0]
                        log_notify(f"[NOTIFY] Closed SELL position, profit/loss: {last_deal.profit}")
                    # Log account balance after reversal to BUY
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after reversing to BUY: {balance}")
                else:
                    log_notify(f"Failed to close SELL and open BUY, retcode = {close_result.retcode}")
        elif trade_signal == "sell":
            if position_type is None:
                # No open position, open SELL
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": bid,
                    "sl": bid + sl_pips,
                    "tp": bid - tp_pips,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "python scalping sell",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] SELL order placed, ticket: {result.order}, price: {bid}")
                    # Log account balance after successful SELL
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after SELL: {balance}")
                else:
                    log_notify(f"SELL order failed, retcode = {result.retcode}")
            elif position_type == 0:
                # Open BUY, but signal is SELL: close BUY, open SELL
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_SELL,
                    "position": ticket,
                    "price": bid,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "close buy, open sell",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                close_result = mt5.order_send(close_request)
                if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] Closed BUY, opened SELL, ticket: {close_result.order}, price: {bid}")
                    # Fetch last deal for profit/loss
                    deals = mt5.history_deals_get(datetime.now(UTC) - timedelta(days=1), datetime.now(UTC))
                    if deals:
                        last_deal = sorted(deals, key=lambda d: d.time, reverse=True)[0]
                        log_notify(f"[NOTIFY] Closed BUY position, profit/loss: {last_deal.profit}")
                    # Log account balance after reversal to SELL
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after reversing to SELL: {balance}")
                else:
                    log_notify(f"Failed to close BUY and open SELL, retcode = {close_result.retcode}")
        else:
            log_notify("No trade signal.")

        # --- Trailing Stop Loss Management (ATR-based) ---
        if enable_trailing_stop:
            # Fetch recent bars for ATR calculation
            atr_bars = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, datetime.now(UTC) - timedelta(minutes=atr_period+2), atr_period+2)
            trailing_pips = 0.0020  # fallback default
            if atr_bars is not None and len(atr_bars) > atr_period:
                atr_df = pd.DataFrame(atr_bars)
                atr_df['high'] = atr_df['high'].astype(float)
                atr_df['low'] = atr_df['low'].astype(float)
                atr_df['close'] = atr_df['close'].astype(float)
                atr_df['prev_close'] = atr_df['close'].shift(1)
                atr_df['tr1'] = atr_df['high'] - atr_df['low']
                atr_df['tr2'] = abs(atr_df['high'] - atr_df['prev_close'])
                atr_df['tr3'] = abs(atr_df['low'] - atr_df['prev_close'])
                atr_df['tr'] = atr_df[['tr1', 'tr2', 'tr3']].max(axis=1)
                atr = atr_df['tr'].rolling(window=atr_period).mean().iloc[-1]
                trailing_pips = atr * atr_multiplier
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    tick = mt5.symbol_info_tick(symbol)
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        new_sl = tick.bid - trailing_pips
                        if (pos.sl is None or pos.sl == 0) or (new_sl > pos.sl):
                            modify_request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": new_sl,
                                "tp": pos.tp,
                                "symbol": symbol,
                                "magic": 234000,
                                "comment": "ATR trailing stop update"
                            }
                            modify_result = mt5.order_send(modify_request)
                            if modify_result and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                                log_notify(f"[TRAILING SL ATR] BUY position {pos.ticket}: SL updated to {new_sl:.5f} (ATR trailing)")
                    elif pos.type == mt5.POSITION_TYPE_SELL:
                        new_sl = tick.ask + trailing_pips
                        if (pos.sl is None or pos.sl == 0) or (new_sl < pos.sl):
                            modify_request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": new_sl,
                                "tp": pos.tp,
                                "symbol": symbol,
                                "magic": 234000,
                                "comment": "ATR trailing stop update"
                            }
                            modify_result = mt5.order_send(modify_request)
                            if modify_result and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
                                log_notify(f"[TRAILING SL ATR] SELL position {pos.ticket}: SL updated to {new_sl:.5f} (ATR trailing)")

        # Wait for 60 seconds before next check
        time.sleep(60)
finally:
    mt5.shutdown()
