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
import csv
import threading
import os
from collections import defaultdict
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from pytz import timezone as ZoneInfo  # fallback for older Python

# ML imports (optional - only needed if using ml_random_forest strategy)
try:
    from ml.model_predictor import ModelPredictor
    from ml.feature_engineering import FeatureEngineering
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] ML modules not available. Install scikit-learn and train model to use ML strategy.")

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

# Define symbol and lot before the main loop (symbol is now configurable via config.json)
# Multi-symbol support: "symbols" list overrides single "symbol" if present
symbols_list = config.get("symbols", [config.get("symbol", "BTCUSD")])
symbol_configs = config.get("symbol_configs", {})
symbol = symbols_list[0]  # Primary symbol for trading
lot = symbol_configs.get(symbol, {}).get("lot", 0.01)

if len(symbols_list) > 1:
    print(f"[INFO] Multi-symbol configured: {symbols_list} (primary: {symbol})")
    print(f"[INFO] Note: ML model currently trained for {symbol} only. Additional symbols require separate models.")

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
    """Log to console, file, AND Telegram (for important trade alerts only)."""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now(UTC).isoformat()} {message}\n")
    send_telegram_message(message)

def log_only(message):
    """Log to console and file only (NO Telegram)."""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now(UTC).isoformat()} {message}\n")

# Hourly heartbeat tracking
last_heartbeat_hour = None

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

def compute_ema(prices, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(prices).ewm(span=period, adjust=False).mean()

def get_ema_trend(prices, fast_period=50, slow_period=200):
    """
    Detect market trend using EMA crossover.
    
    Returns:
        'UPTREND' if EMA50 > EMA200 (bullish)
        'DOWNTREND' if EMA50 < EMA200 (bearish)
        None if not enough data
    """
    if len(prices) < slow_period:
        return None
    
    ema_fast = compute_ema(prices, fast_period).iloc[-1]
    ema_slow = compute_ema(prices, slow_period).iloc[-1]
    
    if ema_fast > ema_slow:
        return "UPTREND"
    else:
        return "DOWNTREND"

# --- ML Model Training automation DAILY at 8:00 AM US Eastern ---
def is_daily_training_time():
    """Check if it's time for daily ML model training (8am-9am ET every day)."""
    try:
        eastern = ZoneInfo("America/New_York")
        now_et = datetime.now(eastern)
    except Exception:
        now_et = datetime.now(UTC) - timedelta(hours=4)
    # Run ML training between 8am-9am ET every day
    return now_et.hour >= 8 and now_et.hour < 9

last_training_date = None
_training_in_progress = False  # Track if background ML training is running
_training_needs_reload = False  # Flag to reload model after background training completes

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

# --- Enhancement Configs ---
min_atr = config.get("min_atr", 0)  # Minimum ATR for volatility filter, default 0 disables filter
news_block_minutes = config.get("news_block_minutes", 15)  # Block trading this many minutes before/after news
higher_timeframe_str = config.get("higher_timeframe", "H1")
higher_timeframe = TIMEFRAME_MAP.get(higher_timeframe_str.upper(), mt5.TIMEFRAME_H1)

# --- Defensive Trading Configs ---
max_spread_percent = config.get("max_spread_percent", 0.05)  # Max spread as % of price (0 = disabled)
loss_cooldown_minutes = config.get("loss_cooldown_minutes", 15)  # Minutes to wait after a loss (0 = disabled)
max_consecutive_losses = config.get("max_consecutive_losses", 3)  # Pause after X consecutive losses (0 = disabled)

# --- EMA Trend Filter Configs ---
enable_ema_trend_filter = config.get("enable_ema_trend_filter", True)  # Enable/disable EMA trend filter
ema_fast_period = config.get("ema_fast_period", 50)  # Fast EMA period
ema_slow_period = config.get("ema_slow_period", 200)  # Slow EMA period

# --- Smart Exit Configs ---
smart_exit_config = config.get("smart_exit", {})
enable_smart_exit = smart_exit_config.get("enabled", False)
max_hold_minutes = smart_exit_config.get("max_hold_minutes", 120)  # Max time to hold a position
close_if_stagnant = smart_exit_config.get("close_if_stagnant", True)
stagnant_threshold_percent = smart_exit_config.get("stagnant_threshold_percent", 0.02)  # 0.02%

# --- News Filter: Forex Factory Economic Calendar ---
_news_cache = {'events': [], 'last_fetch': None}
_NEWS_CACHE_DURATION = timedelta(hours=1)

# Map symbols to currencies affected by economic events
_SYMBOL_CURRENCIES = {
    'BTCUSD': ['USD'],
    'ETHUSD': ['USD'],
    'EURUSD': ['EUR', 'USD'],
    'GBPUSD': ['GBP', 'USD'],
    'USDJPY': ['USD', 'JPY'],
    'XAUUSD': ['USD'],
}

def _fetch_news_calendar():
    """Fetch high-impact economic events from Forex Factory (free, no API key needed)."""
    global _news_cache
    try:
        resp = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10
        )
        if resp.status_code == 200:
            events = resp.json()
            # Filter for high-impact events only
            high_impact = []
            for event in events:
                if event.get('impact', '').lower() == 'high':
                    try:
                        # Parse event date/time (format: "2026-01-27T08:30:00-05:00" or similar)
                        date_str = event.get('date', '')
                        if date_str:
                            event_time = datetime.fromisoformat(date_str)
                            # Convert to UTC if timezone-aware
                            if event_time.tzinfo is not None:
                                event_time = event_time.astimezone(UTC).replace(tzinfo=None)
                            high_impact.append({
                                'time': event_time,
                                'title': event.get('title', 'Unknown'),
                                'country': event.get('country', ''),
                                'currency': event.get('country', '').upper()
                            })
                    except (ValueError, TypeError):
                        continue
            _news_cache['events'] = high_impact
            _news_cache['last_fetch'] = datetime.now(UTC)
            log_only(f"[NEWS] Fetched {len(high_impact)} high-impact events this week")
        else:
            log_only(f"[NEWS] Failed to fetch calendar: HTTP {resp.status_code}")
    except Exception as e:
        log_only(f"[NEWS] Calendar fetch error: {e}")

def is_high_impact_news_near(symbol, block_minutes=15):
    """Check if a high-impact news event is within block_minutes of now."""
    global _news_cache

    # Refresh cache if stale or empty
    if _news_cache['last_fetch'] is None or \
       (datetime.now(UTC) - _news_cache['last_fetch']) > _NEWS_CACHE_DURATION:
        _fetch_news_calendar()

    if not _news_cache['events']:
        return False

    now = datetime.now(UTC).replace(tzinfo=None)
    relevant_currencies = _SYMBOL_CURRENCIES.get(symbol, ['USD'])

    for event in _news_cache['events']:
        # Check if event's currency is relevant to our symbol
        if event['currency'] not in relevant_currencies:
            continue
        # Check if event is within block_minutes of now
        time_diff = abs((event['time'] - now).total_seconds()) / 60
        if time_diff <= block_minutes:
            log_only(f"[NEWS] High-impact event nearby: {event['title']} ({event['currency']}) in {time_diff:.0f} min")
            return True

    return False

# --- Multi-timeframe trend helper ---
def get_higher_tf_trend(symbol, short_ma=10, long_ma=300):
    utc_now = datetime.now(UTC)
    rates = mt5.copy_rates_from(symbol, higher_timeframe, utc_now - timedelta(minutes=long_ma*2), long_ma*2)
    if rates is None or len(rates) < long_ma:
        return None  # Not enough data
    closes = np.array([bar[4] for bar in rates])
    short_ma_val = np.mean(closes[-short_ma:])
    long_ma_val = np.mean(closes[-long_ma:])
    if short_ma_val > long_ma_val:
        return "up"
    elif short_ma_val < long_ma_val:
        return "down"
    else:
        return None

# --- Spread Filter ---
def is_spread_too_wide(symbol, max_percent=0.05):
    """Check if current spread is abnormally wide (sign of low liquidity or manipulation)."""
    if max_percent <= 0:
        return False  # Filter disabled
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.ask == 0:
        return True  # Can't get price, safer to skip
    
    spread = tick.ask - tick.bid
    spread_percent = (spread / tick.ask) * 100
    
    if spread_percent > max_percent:
        return True
    return False

# --- Get correct filling mode for symbol ---
def get_filling_mode(symbol):
    """Get the correct order filling mode for a symbol based on broker support."""
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return mt5.ORDER_FILLING_IOC  # Default fallback
    # filling_mode is a bitmask: 1=FOK supported, 2=IOC supported
    if sym_info.filling_mode & 2:  # IOC supported
        return mt5.ORDER_FILLING_IOC
    elif sym_info.filling_mode & 1:  # FOK supported
        return mt5.ORDER_FILLING_FOK
    else:
        return mt5.ORDER_FILLING_RETURN

# --- Smart Exit Logic ---
def check_smart_exit(symbol, positions, tracked_positions):
    """
    Check if any positions should be closed based on smart exit rules:
    
    1. Time-based exit: Close if position held longer than max_hold_minutes
    2. Stagnation exit: Close if price hasn't moved significantly from entry
    
    Returns: None (closes positions that meet criteria)
    """
    if not enable_smart_exit or not positions:
        return
    
    now = datetime.now(UTC)
    
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue
        
        entry_price = pos.price_open
        current_price = tick.bid if pos.type == 0 else tick.ask
        
        # Calculate how long position has been open
        try:
            # MT5 position time is in seconds since epoch
            position_open_time = datetime.fromtimestamp(pos.time, UTC)
            minutes_held = (now - position_open_time).total_seconds() / 60
        except Exception:
            minutes_held = 0
        
        should_close = False
        close_reason = ""
        
        # Check time-based exit
        if max_hold_minutes > 0 and minutes_held >= max_hold_minutes:
            should_close = True
            close_reason = f"Time limit ({minutes_held:.0f}min > {max_hold_minutes}min)"
        
        # Check stagnation exit
        if not should_close and close_if_stagnant and minutes_held >= 30:  # Only check after 30 min
            price_change_percent = abs((current_price - entry_price) / entry_price) * 100
            if price_change_percent < stagnant_threshold_percent:
                should_close = True
                close_reason = f"Stagnant ({price_change_percent:.3f}% < {stagnant_threshold_percent}%)"
        
        if should_close:
            # Close the position
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if pos.type == 0 else tick.ask
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"smart exit: {close_reason[:20]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": get_filling_mode(symbol),
            }
            
            close_result = mt5.order_send(close_request)
            
            if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                direction = "BUY" if pos.type == 0 else "SELL"
                profit = (close_price - entry_price) * pos.volume if pos.type == 0 else (entry_price - close_price) * pos.volume
                
                log_notify(f"[SMART EXIT] {direction} closed | Reason: {close_reason} | "
                          f"Entry: {entry_price:.2f} | Exit: {close_price:.2f} | "
                          f"Held: {minutes_held:.0f} min")
                
                # Get confidence from tracked positions if available
                confidence = tracked_positions.get(pos.ticket, {}).get('confidence')
                
                # Remove from tracking
                if pos.ticket in tracked_positions:
                    del tracked_positions[pos.ticket]
            else:
                log_only(f"[WARN] Smart exit failed for {pos.ticket}: {close_result.retcode if close_result else 'None'}")

# --- ATR filter helper ---
def get_atr(rates, period=14):
    highs = rates['high']
    lows = rates['low']
    closes = rates['close']
    prev_closes = np.roll(closes, 1)
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
    atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
    return atr

# --- Function to reload configuration after backtest ---
def reload_config_and_strategy():
    """Reload config.json and strategy_params.json after backtest completes."""
    global strategy, config, short_ma_period, long_ma_period, rsi_period
    global macd_fast, macd_slow, macd_signal, bollinger_period, bollinger_stddev

    # Reload config.json to get the new best strategy
    with open("config.json", "r") as f:
        config = json.load(f)
    new_strategy = config.get("strategy", "ma_crossover")

    # Reload strategy_params.json
    with open("strategy_params.json", "r") as f:
        params = json.load(f)

    # Get parameters for the NEW strategy
    strat_params = params.get(new_strategy, {})

    # Update strategy variable
    strategy = new_strategy

    # Update all strategy parameters
    short_ma_period = strat_params.get("short_ma", config.get("short_ma", 10))
    long_ma_period = strat_params.get("long_ma", config.get("long_ma", 300))
    rsi_period = strat_params.get("rsi_period", config.get("rsi_period", 14))
    macd_fast = strat_params.get("macd_fast", config.get("macd_fast", 12))
    macd_slow = strat_params.get("macd_slow", config.get("macd_slow", 26))
    macd_signal = strat_params.get("macd_signal", config.get("macd_signal", 9))
    bollinger_period = strat_params.get("bollinger_period", config.get("bollinger_period", 20))
    bollinger_stddev = strat_params.get("bollinger_stddev", config.get("bollinger_stddev", 2))

    return strategy, strat_params

# --- Initialize ML Predictor (if using ML strategy) ---
ml_predictor = None
ml_feature_eng = None

if strategy == "ml_random_forest":
    if not ML_AVAILABLE:
        print("[ERROR] ML strategy selected but ML modules not available!")
        print("   Install dependencies: pip install scikit-learn joblib")
        sys.exit(1)

    try:
        ml_predictor = ModelPredictor("ml_config.json")
        ml_feature_eng = FeatureEngineering("ml_config.json")
        ml_predictor.load_model()
        print("[ML] Model loaded successfully for ml_random_forest strategy")
    except FileNotFoundError as e:
        print(f"[ERROR] ML model not found: {e}")
        print("   Train the model first: python train_ml_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to initialize ML predictor: {e}")
        sys.exit(1)

# --- Signal handling for graceful shutdown ---
import signal
import os

def shutdown_handler(signum, frame):
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    log_notify(f"[SHUTDOWN] Bot stopped by signal {sig_name} (PID: {os.getpid()})")
    mt5.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# --- Trade Management Loop ---
try:
    # Log bot startup
    account = mt5.account_info()
    startup_balance = account.balance if account else 0
    log_notify(f"[STARTUP] Bot started (PID: {os.getpid()}) | Strategy: {strategy} | Symbol: {symbol} | Balance: ${startup_balance:.2f}")
    log_notify(f"[STARTUP] Config: SL={sl_pips}, TP={tp_pips}, Timeframe={timeframe_str}, Higher TF={higher_timeframe_str}")
    if max_spread_percent > 0 or loss_cooldown_minutes > 0 or max_consecutive_losses > 0:
        log_notify(f"[STARTUP] Defensive: MaxSpread={max_spread_percent}%, Cooldown={loss_cooldown_minutes}min, MaxConsecLoss={max_consecutive_losses}")
    if enable_ema_trend_filter:
        log_notify(f"[STARTUP] EMA Trend Filter: ENABLED (EMA{ema_fast_period}/EMA{ema_slow_period}) - BUY blocked in downtrend")
    if enable_smart_exit:
        log_notify(f"[STARTUP] Smart Exit: ENABLED - Max hold {max_hold_minutes}min, Stagnant check {'ON' if close_if_stagnant else 'OFF'}")
    if strategy == "ml_random_forest":
        log_notify(f"[STARTUP] ML Config: confidence={ml_predictor.confidence_threshold:.0%}, max_hold={ml_predictor.max_hold_probability:.0%}, min_diff={ml_predictor.min_prob_diff:.0%}")

    last_filter_message = None  # Track last filter message to avoid spamming

    # Trade logging setup
    trade_log = []  # In-memory log for current day/week
    trade_log_file = "trade_log.csv"
    daily_pl = 0
    last_pl_date = None

    # Trade statistics tracking
    total_wins = 0
    total_losses = 0
    cumulative_pl = 0.0
    
    # Defensive trading tracking
    consecutive_losses = 0
    last_loss_time = None  # datetime of last losing trade

    # Daily trade limit tracking
    daily_trade_count = 0
    max_trades_per_day = 10  # Default, will be updated from ml_config if available
    if strategy == "ml_random_forest" and ml_predictor is not None:
        max_trades_per_day = ml_predictor.config.get('risk_management', {}).get('max_trades_per_day', 10)
        print(f"[ML] Max trades per day: {max_trades_per_day}")

    # Track open positions to detect SL/TP closures
    tracked_positions = {}  # {ticket: {'direction': 'BUY'/'SELL', 'entry_price': float, 'confidence': float}}

    def append_trade_log(entry):
        trade_log.append(entry)
        with open(trade_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(entry)

    def log_trade_result(direction, entry_price, exit_price, profit, confidence=None, close_reason=""):
        """Log trade result with statistics to both screen and Telegram"""
        global total_wins, total_losses, cumulative_pl, consecutive_losses, last_loss_time

        # Update statistics
        if profit > 0:
            total_wins += 1
            consecutive_losses = 0  # Reset consecutive losses on win
            result = "WIN"
        else:
            total_losses += 1
            consecutive_losses += 1
            last_loss_time = datetime.now(UTC)  # Track when the loss occurred
            result = "LOSS"

        cumulative_pl += profit
        total_trades = total_wins + total_losses
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        # Format trade result message
        conf_str = f" | Confidence: {confidence:.1%}" if confidence else ""
        reason_str = f" ({close_reason})" if close_reason else ""
        trade_msg = f"[TRADE {result}] {direction}{reason_str} | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | P/L: ${profit:.2f}{conf_str}"
        stats_msg = f"[STATS] Wins: {total_wins} | Losses: {total_losses} | Win Rate: {win_rate:.1f}% | Session P/L: ${cumulative_pl:.2f}"

        # Send to both console and Telegram
        log_notify(trade_msg)
        log_notify(stats_msg)
        
        # Alert if consecutive losses threshold reached
        if max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses:
            log_notify(f"[ALERT] {consecutive_losses} consecutive losses! Trading paused for 1 hour.")

    def sync_existing_positions():
        """Sync tracked_positions with actual open positions (for bot restart scenarios)"""
        global tracked_positions

        current_positions = mt5.positions_get(symbol=symbol)
        if not current_positions:
            return

        for pos in current_positions:
            if pos.ticket not in tracked_positions:
                # Found a position we're not tracking - add it
                direction = 'BUY' if pos.type == 0 else 'SELL'
                tracked_positions[pos.ticket] = {
                    'direction': direction,
                    'entry_price': pos.price_open,
                    'confidence': None  # Unknown for existing positions
                }
                log_only(f"[SYNC] Found existing {direction} position, ticket: {pos.ticket}, entry: {pos.price_open:.2f}")

    def check_closed_positions():
        """Check if any tracked positions were closed by SL/TP and log results"""
        global tracked_positions, daily_pl

        # First sync any existing positions we might not be tracking
        sync_existing_positions()

        if not tracked_positions:
            return

        # Get current open positions
        current_positions = mt5.positions_get(symbol=symbol)
        current_tickets = set()
        if current_positions:
            current_tickets = {pos.ticket for pos in current_positions}

        # Check which tracked positions are no longer open
        closed_tickets = set(tracked_positions.keys()) - current_tickets

        for ticket in closed_tickets:
            pos_info = tracked_positions[ticket]
            trade_logged = False

            # Wait for MT5 to sync deal history (broker delay)
            time.sleep(0.3)

            # Get the deal history with retry logic (history may take time to sync)
            closing_deal = None
            for attempt in range(3):
                deals = mt5.history_deals_get(datetime.now(UTC) - timedelta(days=7), datetime.now(UTC))
                if deals:
                    # Find the closing deal for this position
                    for deal in sorted(deals, key=lambda d: d.time, reverse=True):
                        if deal.position_id == ticket and deal.entry == 1:  # entry=1 means exit deal
                            closing_deal = deal
                            break
                if closing_deal:
                    break
                # Wait before retry
                if attempt < 2:
                    time.sleep(0.2)

            if closing_deal:
                exit_price = closing_deal.price
                profit = closing_deal.profit

                # Determine close reason based on deal comment or profit
                close_reason = ""
                if closing_deal.comment:
                    if "sl" in closing_deal.comment.lower() or "stop" in closing_deal.comment.lower():
                        close_reason = "SL Hit"
                    elif "tp" in closing_deal.comment.lower() or "take" in closing_deal.comment.lower():
                        close_reason = "TP Hit"
                    else:
                        close_reason = closing_deal.comment
                else:
                    # Infer from profit if comment not available
                    close_reason = "TP Hit" if profit > 0 else "SL Hit"

                # Log the trade result
                log_trade_result(
                    pos_info['direction'],
                    pos_info['entry_price'],
                    exit_price,
                    profit,
                    pos_info.get('confidence'),
                    close_reason
                )
                append_trade_log([str(datetime.now(UTC)), pos_info['direction'], pos_info['entry_price'], exit_price, profit])
                daily_pl += profit
                check_max_loss_profit()
                trade_logged = True
            else:
                log_notify(f"[WARN] Position {ticket} ({pos_info['direction']}) closed but no exit deal found after 3 retries")

            if not trade_logged:
                # Still log to CSV with estimated data so we don't lose the record
                tick = mt5.symbol_info_tick(symbol)
                est_exit = tick.bid if pos_info['direction'] == 'BUY' else tick.ask if tick else 0
                log_notify(f"[WARN] Logging estimated close for {pos_info['direction']} position {ticket}, entry: {pos_info['entry_price']:.2f}")
                append_trade_log([str(datetime.now(UTC)), pos_info['direction'], pos_info['entry_price'], est_exit, "N/A"])

            # Remove from tracking
            del tracked_positions[ticket]

    while True:
        last_trade_confidence = None  # Reset each iteration
        now = datetime.now(UTC)
        try:
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern)
        except Exception:
            now_et = datetime.now(UTC) - timedelta(hours=4)
        # --- DAILY Backtest automation inside loop ---
        try:
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern)
        except Exception:
            now_et = datetime.now(UTC) - timedelta(hours=4)
        today_date = now_et.date()

        # Reset daily trade counter at day boundary (fixes counter not resetting when no trades occur)
        if last_pl_date is not None and last_pl_date != today_date:
            daily_pl = 0
            daily_trade_count = 0
            last_pl_date = today_date
            log_only(f"[RESET] New trading day ({today_date}) - daily trade count and P/L reset")
        elif last_pl_date is None:
            last_pl_date = today_date

        if is_daily_training_time() and (last_training_date != today_date) and not _training_in_progress:
            log_notify(f"[AUTOMATION] Daily ML training time (8am ET). Starting background training...")
            last_training_date = today_date  # Mark immediately to prevent re-triggering

            def _run_background_training():
                """Run ML model training in a background thread to avoid blocking the trading loop."""
                global _training_in_progress, _training_needs_reload
                _training_in_progress = True
                try:
                    result = subprocess.run(
                        [sys.executable, "train_ml_model.py", "--refresh"],
                        capture_output=True, text=True, timeout=600  # 10 min timeout
                    )
                    print(result.stdout)
                    if result.returncode != 0:
                        log_notify(f"[AUTOMATION] train_ml_model.py failed: {result.stderr[:500]}")
                    else:
                        log_notify("[AUTOMATION] ML model training completed. Will reload on next cycle.")
                        _training_needs_reload = True
                except subprocess.TimeoutExpired:
                    log_notify("[AUTOMATION] ML training timed out after 10 minutes")
                except Exception as e:
                    log_notify(f"[AUTOMATION] Background training error: {e}")
                finally:
                    _training_in_progress = False

            training_thread = threading.Thread(target=_run_background_training, daemon=True)
            training_thread.start()

        # Check if background training finished and model needs reloading
        if _training_needs_reload and not _training_in_progress:
            try:
                if ml_predictor is not None:
                    ml_predictor.load_model()
                log_notify(f"[AUTOMATION] ML model reloaded successfully after background training")
            except Exception as e:
                log_notify(f"[AUTOMATION] Failed to reload ML model: {e}")
            _training_needs_reload = False
        # (Re)initialize and ensure the symbol is available
        if not mt5.initialize(login=login, password=password, server=server):
            log_only(f"[ERROR] MT5 initialize() failed, error code={mt5.last_error()}")
            time.sleep(60)
            continue
        if not mt5.symbol_select(symbol, True):
            log_only(f"[ERROR] Failed to select symbol {symbol}. Waiting...")
            time.sleep(60)
            continue

        # Check if any tracked positions were closed by SL/TP
        check_closed_positions()
        
        # Check for smart exit conditions (time-based, stagnation)
        current_positions = mt5.positions_get(symbol=symbol)
        if enable_smart_exit and current_positions:
            check_smart_exit(symbol, current_positions, tracked_positions)

        # Check if market is open for trading
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            log_only(f"[ERROR] symbol_info for {symbol} is None. Waiting...")
            time.sleep(60)
            continue
        market_open = (
            symbol_info.visible and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
        )
        if not market_open:
            log_only(f"[HEARTBEAT] Market is closed for {symbol}. Bot is running. UTC: {now.isoformat()}, ET: {now_et.isoformat()}")
            time.sleep(60)
            continue

        # --- Hourly heartbeat to Telegram ---
        current_hour = now_et.hour
        if last_heartbeat_hour != current_hour:
            account = mt5.account_info()
            balance = account.balance if account else 0
            log_notify(f"[HEARTBEAT] Bot running. Balance: ${balance:.2f} | Strategy: {strategy} | {symbol}")
            last_heartbeat_hour = current_hour

        # Fetch the last 250 bars for the configured symbol
        utc_now = datetime.now(UTC)
        rates = mt5.copy_rates_from(symbol, timeframe, utc_now - timedelta(minutes=250), 250)
        if rates is None or len(rates) == 0:
            log_only(f"Failed to get bars for {symbol}. Waiting...")
            time.sleep(60)
            continue
        closes = np.array([bar[4] for bar in rates])
        # --- Strategy selection ---
        trade_signal = None
        # Multi-timeframe trend filter
        higher_tf_trend = get_higher_tf_trend(symbol, short_ma_period, long_ma_period)
        if higher_tf_trend is None:
            msg = "[FILTER] Not enough higher timeframe data for trend confirmation. No trade."
            if last_filter_message != msg:
                log_only(msg)
                last_filter_message = msg
            continue
        # ATR filter
        atr_val = get_atr(rates, atr_period)
        if atr_val < min_atr:
            msg = f"[FILTER] ATR {atr_val:.5f} below threshold {min_atr}. No trade."
            if last_filter_message != msg:
                log_only(msg)
                last_filter_message = msg
            continue
        # News filter
        if is_high_impact_news_near(symbol, news_block_minutes):
            msg = f"[FILTER] High-impact news event near. No trade."
            if last_filter_message != msg:
                log_only(msg)
                last_filter_message = msg
            continue
        
        # EMA Trend Filter - detect overall market trend
        ema_trend = None
        if enable_ema_trend_filter:
            ema_trend = get_ema_trend(closes, ema_fast_period, ema_slow_period)
            if ema_trend:
                ema_fast_val = compute_ema(closes, ema_fast_period).iloc[-1]
                ema_slow_val = compute_ema(closes, ema_slow_period).iloc[-1]
                log_only(f"[TREND] EMA{ema_fast_period}: {ema_fast_val:.2f} | EMA{ema_slow_period}: {ema_slow_val:.2f} | Trend: {ema_trend}")
        
        last_filter_message = None  # Reset if all filters pass
        # --- Strategy logic as before, but only take trade if M5 and H1 agree ---
        if strategy == "ma_crossover":
            short_ma = np.mean(closes[-short_ma_period:])
            long_ma = np.mean(closes[-long_ma_period:])
            if short_ma > long_ma and higher_tf_trend == "up":
                trade_signal = "buy"
            elif short_ma < long_ma and higher_tf_trend == "down":
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
        elif strategy == "ml_random_forest":
            # Machine Learning strategy using trained Random Forest model
            if ml_predictor is not None and ml_feature_eng is not None:
                try:
                    # Calculate all features from current market data
                    df_temp = pd.DataFrame(rates)
                    df_temp['timestamp'] = pd.to_datetime(df_temp['time'], unit='s')
                    df_temp.rename(columns={
                        'tick_volume': 'volume'
                    }, inplace=True)

                    # Add features
                    df_with_features = ml_feature_eng.add_all_features(df_temp)

                    # Get latest features (last row)
                    latest_features = {}
                    feature_cols = ml_feature_eng.get_feature_columns()
                    for feat in feature_cols:
                        if feat in df_with_features.columns:
                            latest_features[feat] = df_with_features[feat].iloc[-1]

                    # VOLATILITY FILTER: Check ATR threshold
                    current_atr = latest_features.get('atr_14', 0)
                    min_atr_threshold = ml_predictor.config.get('risk_management', {}).get('min_atr_threshold', 50)

                    if current_atr < min_atr_threshold:
                        msg = f"[ML FILTER] ATR {current_atr:.1f} below threshold {min_atr_threshold}. Skipping trade."
                        if last_filter_message != msg:
                            log_only(msg)
                            last_filter_message = msg
                        trade_signal = None
                        continue

                    # Get ML prediction
                    signal, confidence, reason = ml_predictor.get_trade_signal(latest_features)

                    if signal is not None:
                        trade_signal = signal
                        last_trade_confidence = confidence  # Store for position tracking
                        log_only(f"[ML] {reason} | ATR: {current_atr:.1f} | Probabilities: " +
                                ", ".join([f"{k}:{v:.1%}" for k, v in
                                          ml_predictor.predict(latest_features)[2].items()]))
                        last_filter_message = None  # Reset filter message
                    else:
                        trade_signal = None
                        msg = f"[ML] {reason}"
                        if last_filter_message != msg:
                            log_only(msg)
                            last_filter_message = msg

                except Exception as e:
                    log_only(f"[ML ERROR] Failed to get prediction: {e}")
                    trade_signal = None
            else:
                log_only("[ML ERROR] ML predictor not initialized")
                trade_signal = None
        elif strategy == "custom":
            log_only("Custom strategy not implemented. Please add your logic.")
            trade_signal = None
        else:
            log_only(f"Unknown strategy: {strategy}. No trades will be made.")
            trade_signal = None
        # log_only(debug_msg)  # Commented out for future troubleshooting
        # log_only(f"[DEBUG] trade_signal={trade_signal}")  # Commented out for future troubleshooting

        # --- EMA TREND FILTER: Block trades against the trend ---
        if trade_signal is not None and enable_ema_trend_filter and ema_trend is not None:
            if trade_signal == "buy" and ema_trend == "DOWNTREND":
                msg = f"[TREND FILTER] BUY signal blocked - Market in DOWNTREND (EMA{ema_fast_period} < EMA{ema_slow_period})"
                if last_filter_message != msg:
                    log_only(msg)
                    last_filter_message = msg
                trade_signal = None  # Block the BUY trade in downtrend
            elif trade_signal == "sell" and ema_trend == "UPTREND":
                msg = f"[TREND FILTER] SELL signal blocked - Market in UPTREND (EMA{ema_fast_period} > EMA{ema_slow_period})"
                if last_filter_message != msg:
                    log_only(msg)
                    last_filter_message = msg
                trade_signal = None  # Block the SELL trade in uptrend

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log_only(f"Failed to get current tick for {symbol}. Waiting...")
            time.sleep(60)
            continue
        ask = tick.ask
        bid = tick.bid
        if ask == 0 or bid == 0:
            log_only("Invalid ask/bid price. Waiting...")
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

        # Check daily trade limit before executing any new trades
        if trade_signal is not None and daily_trade_count >= max_trades_per_day:
            msg = f"[LIMIT] Max daily trades reached ({daily_trade_count}/{max_trades_per_day}) - no new trades"
            if last_filter_message != msg:
                log_only(msg)
                last_filter_message = msg
            trade_signal = None  # Block the trade

        # --- DEFENSIVE TRADING FILTERS ---
        
        # Check spread filter
        if trade_signal is not None and max_spread_percent > 0:
            if is_spread_too_wide(symbol, max_spread_percent):
                tick = mt5.symbol_info_tick(symbol)
                spread = tick.ask - tick.bid if tick else 0
                spread_pct = (spread / tick.ask * 100) if tick and tick.ask > 0 else 0
                msg = f"[FILTER] Spread too wide: {spread:.2f} ({spread_pct:.3f}%) > {max_spread_percent}% - no trade"
                if last_filter_message != msg:
                    log_only(msg)
                    last_filter_message = msg
                trade_signal = None
        
        # Check loss cooldown
        if trade_signal is not None and loss_cooldown_minutes > 0 and last_loss_time is not None:
            minutes_since_loss = (datetime.now(UTC) - last_loss_time).total_seconds() / 60
            if minutes_since_loss < loss_cooldown_minutes:
                remaining = loss_cooldown_minutes - minutes_since_loss
                msg = f"[COOLDOWN] {remaining:.1f} min remaining after last loss - no trade"
                if last_filter_message != msg:
                    log_only(msg)
                    last_filter_message = msg
                trade_signal = None
        
        # Check consecutive losses circuit breaker
        if trade_signal is not None and max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses:
            msg = f"[CIRCUIT BREAKER] {consecutive_losses} consecutive losses - trading paused"
            if last_filter_message != msg:
                log_only(msg)
                last_filter_message = msg
            # Reset after 1 hour of pause
            if last_loss_time is not None:
                hours_since_loss = (datetime.now(UTC) - last_loss_time).total_seconds() / 3600
                if hours_since_loss >= 1:
                    consecutive_losses = 0
                    log_notify(f"[CIRCUIT BREAKER] Reset after 1 hour cooldown. Trading resumed.")
            trade_signal = None

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
                    "type_filling": get_filling_mode(symbol),
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] BUY order placed, ticket: {result.order}, price: {ask}")
                    # Track this position for SL/TP detection
                    ml_conf = last_trade_confidence if strategy == "ml_random_forest" else None
                    tracked_positions[result.order] = {
                        'direction': 'BUY',
                        'entry_price': ask,
                        'confidence': ml_conf
                    }
                    # Log account balance after successful BUY
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after BUY: ${balance:.2f}")
                    daily_trade_count += 1  # Increment daily trade counter
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
                    "type_filling": get_filling_mode(symbol),
                }
                close_result = mt5.order_send(close_request)
                if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] Closed SELL, opened BUY, ticket: {close_result.order}, price: {ask}")
                    # Remove old position from tracking
                    if ticket in tracked_positions:
                        old_conf = tracked_positions[ticket].get('confidence')
                        del tracked_positions[ticket]
                    else:
                        old_conf = None
                    # Fetch last deal for profit/loss
                    deals = mt5.history_deals_get(datetime.now(UTC) - timedelta(days=1), datetime.now(UTC))
                    if deals:
                        last_deal = sorted(deals, key=lambda d: d.time, reverse=True)[0]
                        # Log trade result with statistics
                        log_trade_result("SELL", entry_price, tick.bid, last_deal.profit, old_conf, "Reversal")
                        append_trade_log([str(datetime.now(UTC)), "SELL", entry_price, tick.bid, last_deal.profit])
                        daily_pl += last_deal.profit
                        check_max_loss_profit()
                    # Track the new BUY position
                    ml_conf = last_trade_confidence if strategy == "ml_random_forest" else None
                    tracked_positions[close_result.order] = {
                        'direction': 'BUY',
                        'entry_price': ask,
                        'confidence': ml_conf
                    }
                    # Log account balance after reversal to BUY
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after reversing to BUY: ${balance:.2f}")
                    daily_trade_count += 1  # Increment daily trade counter
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
                    "type_filling": get_filling_mode(symbol),
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] SELL order placed, ticket: {result.order}, price: {bid}")
                    # Track this position for SL/TP detection
                    ml_conf = last_trade_confidence if strategy == "ml_random_forest" else None
                    tracked_positions[result.order] = {
                        'direction': 'SELL',
                        'entry_price': bid,
                        'confidence': ml_conf
                    }
                    # Log account balance after successful SELL
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after SELL: ${balance:.2f}")
                    daily_trade_count += 1  # Increment daily trade counter
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
                    "type_filling": get_filling_mode(symbol),
                }
                close_result = mt5.order_send(close_request)
                if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                    log_notify(f"[NOTIFY] Closed BUY, opened SELL, ticket: {close_result.order}, price: {bid}")
                    # Remove old position from tracking
                    if ticket in tracked_positions:
                        old_conf = tracked_positions[ticket].get('confidence')
                        del tracked_positions[ticket]
                    else:
                        old_conf = None
                    # Fetch last deal for profit/loss
                    deals = mt5.history_deals_get(datetime.now(UTC) - timedelta(days=1), datetime.now(UTC))
                    if deals:
                        last_deal = sorted(deals, key=lambda d: d.time, reverse=True)[0]
                        # Log trade result with statistics
                        log_trade_result("BUY", entry_price, tick.ask, last_deal.profit, old_conf, "Reversal")
                        append_trade_log([str(datetime.now(UTC)), "BUY", entry_price, tick.ask, last_deal.profit])
                        daily_pl += last_deal.profit
                        check_max_loss_profit()
                    # Track the new SELL position
                    ml_conf = last_trade_confidence if strategy == "ml_random_forest" else None
                    tracked_positions[close_result.order] = {
                        'direction': 'SELL',
                        'entry_price': bid,
                        'confidence': ml_conf
                    }
                    # Log account balance after reversal to SELL
                    balance = mt5.account_info().balance
                    log_notify(f"[BALANCE] Account balance after reversing to SELL: ${balance:.2f}")
                    daily_trade_count += 1  # Increment daily trade counter
                else:
                    log_notify(f"Failed to close BUY and open SELL, retcode = {close_result.retcode}")
        else:
            log_only("No trade signal.")

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

        # --- Trade summary and critical alert enhancements ---
        max_daily_loss = config.get("max_daily_loss", 100)  # USD, set in config.json
        max_daily_profit = config.get("max_daily_profit", 200)  # USD, set in config.json
        last_summary_date = None
        last_week_number = None

        def send_trade_summary(period="daily"):
            if not trade_log:
                return
            # Filter out entries with non-numeric profit (like "N/A") to avoid type error
            numeric_profits = [t[4] for t in trade_log if isinstance(t[4], (int, float))]
            total_profit = sum(numeric_profits) if numeric_profits else 0
            num_trades = len(trade_log)
            buys = sum(1 for t in trade_log if t[1] == "BUY")
            sells = sum(1 for t in trade_log if t[1] == "SELL")
            msg = f"[SUMMARY] {period.capitalize()} Trade Summary:\nTotal Trades: {num_trades}\nBuys: {buys}, Sells: {sells}\nTotal P/L: {total_profit:.2f} USD"
            send_telegram_message(msg)
            # Optionally, email or other notification here

        def check_and_send_summaries():
            global last_summary_date, last_week_number, trade_log
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern)
            today = now_et.date()
            week = today.isocalendar()[1]
            # Daily summary at 5pm ET
            if last_summary_date != today and now_et.hour >= 17:
                send_trade_summary("daily")
                trade_log = []  # Reset for new day
                last_summary_date = today
            # Weekly summary on Friday after 5pm ET
            if now_et.weekday() == 4 and last_week_number != week and now_et.hour >= 17:
                send_trade_summary("weekly")
                last_week_number = week

        def check_max_loss_profit():
            global daily_pl, last_pl_date, daily_trade_count
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern)
            today = now_et.date()
            if last_pl_date != today:
                daily_pl = 0
                daily_trade_count = 0  # Reset daily trade count
                last_pl_date = today
                log_only(f"[RESET] New trading day - trade count reset")
            if abs(daily_pl) >= max_daily_loss:
                send_telegram_message(f"[ALERT] Max daily loss reached: {daily_pl:.2f} USD. Trading paused.")
                time.sleep(3600)  # Pause for 1 hour
            if daily_pl >= max_daily_profit:
                send_telegram_message(f"[ALERT] Max daily profit reached: {daily_pl:.2f} USD. Trading paused.")
                time.sleep(3600)  # Pause for 1 hour

        # Wait for 60 seconds before next check
        time.sleep(60)
        check_and_send_summaries()
except Exception as e:
    log_notify(f"[SHUTDOWN] Bot crashed with error: {e} (PID: {os.getpid()})")
    raise
finally:
    try:
        account = mt5.account_info()
        final_balance = account.balance if account else 0
        log_notify(f"[SHUTDOWN] Bot shutting down (PID: {os.getpid()}) | Final Balance: ${final_balance:.2f}")
    except Exception:
        log_only(f"[SHUTDOWN] Bot shutting down (PID: {os.getpid()}) | Balance unavailable")
    mt5.shutdown()
