import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import calendar
import time
import json

# Read credentials and SL/TP from config.json
with open("config.json", "r") as f:
    config = json.load(f)
login = config["login"]
password = config["password"]
server = config["server"]
sl_pips = config.get("sl_pips", 0.0010)  # default 10 pips
tp_pips = config.get("tp_pips", 0.0020)  # default 20 pips

# Connect to MT5 (default path or existing instance)
if not mt5.initialize(login=login, password=password, server=server):
    print("MT5 initialize() failed, error code=", mt5.last_error())
    mt5.shutdown()
    quit()
print("Connected to MT5:", mt5.terminal_info().name)
# Get account information
account_info = mt5.account_info()

symbol = "EURUSD"
lot = 0.01

# Load optimized parameters if available
try:
    with open("strategy_params.json", "r") as f:
        params = json.load(f)
    short_ma_period = params.get("short_ma", 10)
    long_ma_period = params.get("long_ma", 300)
except Exception:
    short_ma_period = 10
    long_ma_period = 300

# Check if today is weekend
now = datetime.now(UTC)
if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
    print("The market is closed on weekends. Please try again during weekdays.")
    mt5.shutdown()
    quit()

# (Re)initialize and ensure the symbol is available
mt5.initialize(login=login, password=password, server=server)
mt5.symbol_select(symbol, True)

# Check if market is open for trading
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None or not symbol_info.visible:
    print(f"Symbol {symbol} is not available for trading.")
    mt5.shutdown()
    quit()
# Check if trading is allowed (trade_mode == SYMBOL_TRADE_MODE_FULL)
if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
    print(f"Market is currently closed for {symbol} (trade_mode={symbol_info.trade_mode}). No trades will be sent.")
    mt5.shutdown()
    quit()

# Fetch the last 250 1-minute bars for EURUSD
utc_now = datetime.now(UTC)
rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_now - timedelta(minutes=250), 250)
if rates is None or len(rates) == 0:
    print("Failed to get bars for", symbol)
    mt5.shutdown()
    quit()

# Compute moving averages (simple MA)
closes = np.array([bar[4] for bar in rates])  # index 4 = close price
short_ma = np.mean(closes[-short_ma_period:])   # optimized short MA
long_ma  = np.mean(closes[-long_ma_period:])    # optimized long MA

# Get current price
tick = mt5.symbol_info_tick(symbol)
if tick is None:
    print("Failed to get current tick for", symbol)
    mt5.shutdown()
    quit()
ask = tick.ask
bid = tick.bid

# Ensure prices are valid
if ask == 0 or bid == 0:
    print("Invalid ask/bid price.")
    mt5.shutdown()
    quit()

# Determine signal and send order
if short_ma > long_ma:
    # Bullish signal: place a BUY order
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": ask,
        "sl": ask - sl_pips,    # example 10-pip stop loss
        "tp": ask + tp_pips,    # example 20-pip take profit
        "deviation": 20,
        "magic": 234000,
        "comment": "python scalping buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print("BUY order placed, ticket:", result.order)
    else:
        print("BUY order failed, retcode =", result.retcode)
elif short_ma < long_ma:
    # Bearish signal: place a SELL order
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
    # Check if price is valid for SELL
    if bid > 0:
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print("SELL order placed, ticket:", result.order)
        else:
            print("SELL order failed, retcode =", result.retcode)
    else:
        print("SELL order not sent: invalid bid price.")

mt5.shutdown()


positions = mt5.positions_get(symbol="EURUSD")
if positions:
    for pos in positions:
        print(pos)  # shows ticket, type, volume, profit, etc.


log_file = "trade_notifications.log"

def log_notify(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now(UTC).isoformat()} {message}\n")

# --- Trade Management Loop ---
try:
    while True:
        now = datetime.now(UTC)
        # (Re)initialize and ensure the symbol is available
        mt5.initialize(login=login, password=password, server=server)
        mt5.symbol_select(symbol, True)

        # Check if market is open for trading
        symbol_info = mt5.symbol_info(symbol)
        market_open = (
            symbol_info is not None and symbol_info.visible and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
        )
        if not market_open:
            log_notify(f"Market is closed for {symbol}. Waiting...")
            time.sleep(60)
            continue

        # Fetch the last 250 1-minute bars for EURUSD
        utc_now = datetime.now(UTC)
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_now - timedelta(minutes=250), 250)
        if rates is None or len(rates) == 0:
            log_notify(f"Failed to get bars for {symbol}. Waiting...")
            time.sleep(60)
            continue
        # Compute moving averages (simple MA)
        closes = np.array([bar[4] for bar in rates])
        short_ma = np.mean(closes[-short_ma_period:])
        long_ma  = np.mean(closes[-long_ma_period:])

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
        if short_ma > long_ma:
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
                else:
                    log_notify(f"Failed to close SELL and open BUY, retcode = {close_result.retcode}")
        elif short_ma < long_ma:
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
                else:
                    log_notify(f"Failed to close BUY and open SELL, retcode = {close_result.retcode}")
        else:
            log_notify("No trade signal.")

        # --- Trailing Stop Loss Management ---
        trailing_pips = 0.0020  # 20 pips for EURUSD
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                # Get latest price
                tick = mt5.symbol_info_tick(symbol)
                if pos.type == mt5.POSITION_TYPE_BUY:
                    # For BUY, trail SL up as price rises
                    new_sl = tick.bid - trailing_pips
                    # Only move SL up (never down)
                    if (pos.sl is None or pos.sl == 0) or (new_sl > pos.sl):
                        modify_result = mt5.order_modify(pos.ticket, pos.price_open, new_sl, pos.tp, 0)
                        if modify_result:
                            log_notify(f"[TRAILING SL] BUY position {pos.ticket}: SL updated to {new_sl:.5f}")
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    # For SELL, trail SL down as price falls
                    new_sl = tick.ask + trailing_pips
                    # Only move SL down (never up)
                    if (pos.sl is None or pos.sl == 0) or (new_sl < pos.sl):
                        modify_result = mt5.order_modify(pos.ticket, pos.price_open, new_sl, pos.tp, 0)
                        if modify_result:
                            log_notify(f"[TRAILING SL] SELL position {pos.ticket}: SL updated to {new_sl:.5f}")

        # Wait for 60 seconds before next check
        time.sleep(60)
finally:
    mt5.shutdown()
