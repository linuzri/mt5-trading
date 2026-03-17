import MetaTrader5 as mt5
import json

with open("mt5_auth.json", "r") as f:
    auth = json.load(f)

with open("config.json", "r") as f:
    config = json.load(f)

if not mt5.initialize(login=auth["login"], password=auth["password"], server=auth["server"]):
    print(f"Failed: {mt5.last_error()}")
    quit()

# Symbol is now configurable via config.json
symbol = config.get("symbol", "BTCUSD")
mt5.symbol_select(symbol, True)
tick = mt5.symbol_info_tick(symbol)

print(f"Current Bid: {tick.bid}, Ask: {tick.ask}")

# Get the correct filling mode for this symbol
symbol_info = mt5.symbol_info(symbol)
print(f"Symbol filling_mode value: {symbol_info.filling_mode}")

# filling_mode is a bitmask: 1=FOK, 2=IOC
# If filling_mode = 2, use IOC (ORDER_FILLING_IOC = 1)
filling_type = mt5.ORDER_FILLING_IOC
print(f"Using filling mode: {filling_type} (IOC)")

# Test order WITHOUT SL/TP first
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_SELL,
    "price": tick.bid,
    "deviation": 20,
    "magic": 234000,
    "comment": "test order",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": filling_type,
}

print("\nTrying order WITHOUT SL/TP...")
result = mt5.order_send(request)
print(f"Result: {result}")
print(f"Retcode: {result.retcode}")

if result.retcode == mt5.TRADE_RETCODE_DONE:
    print("[OK] Order successful! Closing it now...")
    # Get fresh tick for closing
    tick = mt5.symbol_info_tick(symbol)
    # Close the position
    close = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "position": result.order,
        "price": tick.ask,
        "deviation": 20,
        "magic": 234000,
        "type_filling": filling_type,
    })
    print(f"Close result: {close.retcode}")
else:
    print(f"[X] Order failed with retcode: {result.retcode}")
    print(f"    Comment: {result.comment}")

mt5.shutdown()
