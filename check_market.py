import MetaTrader5 as mt5
import json
from datetime import datetime

with open("mt5_auth.json", "r") as f:
    auth = json.load(f)

with open("config.json", "r") as f:
    config = json.load(f)

if not mt5.initialize(login=auth["login"], password=auth["password"], server=auth["server"]):
    print(f"Failed: {mt5.last_error()}")
    quit()

# Symbol is now configurable via config.json
symbol = config.get("symbol", "BTCUSD")

print(f"=== Current Time ===")
print(f"Your PC time: {datetime.now()}")
tick = mt5.symbol_info_tick(symbol)
if tick:
    print(f"Server time: {datetime.fromtimestamp(tick.time)}")
mt5.symbol_select(symbol, True)
info = mt5.symbol_info(symbol)

print(f"\n=== Market Status ===")
print(f"Symbol: {symbol}")
print(f"Visible: {info.visible}")
print(f"Trade Mode: {info.trade_mode}")
print(f"  0 = DISABLED")
print(f"  1 = LONGONLY")
print(f"  2 = SHORTONLY") 
print(f"  3 = CLOSEONLY")
print(f"  4 = FULL (should allow trading)")

# Get account info
account = mt5.account_info()
if account:
    print(f"\n=== Account Info ===")
    print(f"Login: {account.login}")
    print(f"Trade Allowed: {account.trade_allowed}")
    print(f"Trade Expert: {account.trade_expert}")
    print(f"Balance: {account.balance}")
    print(f"Leverage: {account.leverage}")
    print(f"Server: {account.server}")

# Check trading session
print(f"\n=== Trading Session ===")
print(f"Session Open: {info.session_open}")
print(f"Session Close: {info.session_close}")

# Check if orders are allowed
print(f"\n=== Order Modes Allowed ===")
print(f"Order Mode: {info.order_mode} (binary)")
print(f"  Bit 0 (1) = Market orders")
print(f"  Bit 1 (2) = Limit orders")
print(f"  Bit 2 (4) = Stop orders")
print(f"  Bit 3 (8) = Stop Limit orders")
print(f"  127 = All order types allowed")

# Get terminal info
terminal = mt5.terminal_info()
print(f"\n=== Terminal Info ===")
print(f"Connected: {terminal.connected}")
print(f"Trade Allowed: {terminal.trade_allowed}")
print(f"Company: {terminal.company}")

print(f"\n=== Diagnosis ===")
if info.trade_mode != 4:
    print("[X] Trading is restricted on this symbol")
elif not account.trade_allowed:
    print("[X] Trading is disabled on your account")
elif not terminal.trade_allowed:
    print("[X] AutoTrading is disabled in MT5 terminal - Click the 'AutoTrading' button!")
elif not terminal.connected:
    print("[X] Terminal is not connected to the broker")
else:
    print("[OK] All settings look correct - Trading should be available")
    if symbol.startswith("BTC") or symbol.startswith("ETH") or symbol.startswith("XBT"):
        print("[i] Crypto markets trade 24/7")
    else:
        print("[!] If Forex, market is likely CLOSED (weekend/holiday/off-hours)")
        print(f"    Forex market: Sunday 5pm EST - Friday 5pm EST")

mt5.shutdown()
