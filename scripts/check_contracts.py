import MetaTrader5 as mt5
mt5.initialize()

for sym in ['BTCUSD', 'XAUUSD', 'EURUSD']:
    info = mt5.symbol_info(sym)
    if info:
        pip_value = info.trade_tick_value / info.trade_tick_size if info.trade_tick_size > 0 else 1.0
        print(f"\n=== {sym} ===")
        print(f"  Contract size: {info.trade_contract_size}")
        print(f"  Tick value: {info.trade_tick_value}")
        print(f"  Tick size: {info.trade_tick_size}")
        print(f"  Pip value (tick_value/tick_size): {pip_value}")
        print(f"  Volume min: {info.volume_min} / max: {info.volume_max} / step: {info.volume_step}")
        
        # Calculate: with max_lot and SL, what's the risk?
        configs = {'BTCUSD': (100, 0.05), 'XAUUSD': (40, 0.02), 'EURUSD': (15, 0.05)}
        sl, max_lot = configs.get(sym, (40, 0.05))
        old_risk = max_lot * sl  # old formula
        new_risk = max_lot * sl * pip_value  # actual risk
        
        balance = 49430
        risk_pct = 0.5
        risk_amt = balance * risk_pct / 100
        correct_lot = risk_amt / (sl * pip_value) if sl * pip_value > 0 else 0
        
        print(f"  SL config: {sl} pips | Max lot: {max_lot}")
        print(f"  Old formula risk (lot*SL): ${old_risk:.2f}")
        print(f"  Actual risk (lot*SL*pip_value): ${new_risk:.2f}")
        print(f"  Correct lot for ${risk_amt:.2f} risk: {correct_lot:.4f}")
    else:
        print(f"\n{sym}: Not found")

mt5.shutdown()
