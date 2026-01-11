import MetaTrader5 as mt5
import json

# Read credentials
with open("mt5_auth.json", "r") as f:
    auth = json.load(f)

# Read config
with open("config.json", "r") as f:
    config = json.load(f)

# Connect
if not mt5.initialize(login=auth["login"], password=auth["password"], server=auth["server"]):
    print(f"Failed: {mt5.last_error()}")
    quit()

# Symbol is now configurable via config.json
symbol = config.get("symbol", "BTCUSD")
mt5.symbol_select(symbol, True)
info = mt5.symbol_info(symbol)

if info:
    print(f"=== {symbol} Symbol Information ===\n")
    
    # Convert to dict to see all attributes
    info_dict = info._asdict()
    
    # Print key attributes
    print(f"Point: {info.point}")
    print(f"Digits: {info.digits}")
    print(f"Spread: {info.spread}")
    print(f"Trade Contract Size: {info.trade_contract_size}")
    print(f"Volume Min: {info.volume_min}")
    print(f"Volume Max: {info.volume_max}")
    print(f"Volume Step: {info.volume_step}")
    
    # Check for stops-related attributes
    print("\n=== Stops & Limits ===")
    for key, value in info_dict.items():
        if 'stop' in key.lower() or 'freeze' in key.lower() or 'level' in key.lower():
            print(f"{key}: {value}")
    
    # Get current tick to show bid/ask
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"\n=== Current Price ===")
        print(f"Bid: {tick.bid}")
        print(f"Ask: {tick.ask}")
        print(f"Spread: {tick.ask - tick.bid:.5f}")
    
    # Print ALL attributes (for debugging)
    print("\n=== ALL Symbol Attributes ===")
    for key, value in sorted(info_dict.items()):
        print(f"{key}: {value}")
        
else:
    print(f"Failed to get symbol info")

mt5.shutdown()
