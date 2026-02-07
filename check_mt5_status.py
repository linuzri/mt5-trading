import json, MetaTrader5 as mt5, os, sys
auth_path = r"C:\Users\Nazri Hussain\Projects\mt5-trading\mt5-trading\mt5_auth.json"
with open(auth_path) as f:
    auth = json.load(f)
if not mt5.initialize(login=auth.get('login'), password=auth.get('password'), server=auth.get('server')):
    print('INIT_FAIL', mt5.last_error())
    sys.exit(1)
acct = mt5.account_info()
if acct:
    print(f'BALANCE:{acct.balance}')
else:
    print('NO_ACCOUNT')
positions = mt5.positions_get(symbol='BTCUSD')
if positions:
    for p in positions:
        typ = 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL'
        print(f'POS:{p.ticket}|type:{typ}|entry:{p.price_open:.5f}|vol:{p.volume}')
else:
    print('NO_POS')
mt5.shutdown()
