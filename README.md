# MT5-Trading Python Bot

A robust, fully automated MetaTrader 5 (MT5) trading bot in Python supporting multiple strategies, advanced risk management, and comprehensive notifications/logging. Supports both **Forex and Crypto CFDs** (BTCUSD, ETHUSD, etc.).

## Features

### Core Trading
- **Automated trading on MT5** (Pepperstone, IC Markets, or any MT5 broker)
- **Configurable symbol** - Trade Forex (EURUSD) or Crypto (BTCUSD, ETHUSD)
- **Multiple strategies:** MA Crossover, RSI, MACD, Bollinger Bands
- **Scalping support:** M1 timeframe for quick trades
- **All settings configurable** via `config.json`

### Smart Optimization
- **Daily auto-optimization** - Runs every day at 8:00 AM US Eastern
- **Auto-select best strategy** - Automatically switches to the most profitable strategy
- **Multi-timeframe confirmation** - Only trades when lower and higher timeframes agree
- **ATR volatility filter** - Trades only when volatility is above threshold

### Risk Management
- **Configurable SL/TP** - Stop loss and take profit in price units
- **Trailing stop loss** - ATR-based trailing stop management
- **Daily loss/profit limits** - Auto-pause when limits reached
- **News event avoidance** - Placeholder for news API integration

### Notifications
- **Telegram alerts** - Only for trades (BUY/SELL), not spam
- **Hourly heartbeat** - Confirms bot is running with balance info
- **Trade summaries** - Daily and weekly P/L summaries
- **Critical alerts** - Max loss/profit triggers immediate notification

### Logging
- **Console output** - Real-time status
- **File logging** - All events to `trade_notifications.log`
- **Trade history** - Closed trades to `trade_log.csv`

## Requirements

- Python 3.8+
- MetaTrader 5 Terminal (with AutoTrading enabled)
- MT5 Broker account (demo or live)

```sh
pip install MetaTrader5 pandas numpy requests pytz
```

## Quick Start

### 1. Clone and Install
```sh
git clone https://github.com/your-repo/mt5-trading.git
cd mt5-trading
pip install -r requirements.txt
```

### 2. Create `mt5_auth.json`
```json
{
  "login": 123456789,
  "password": "YOUR_MT5_PASSWORD",
  "server": "Pepperstone-Demo",
  "telegram": {
    "bot_name": "@your_telegram_bot",
    "api_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "chat_id": 123456789
  }
}
```

### 3. Configure `config.json`
```json
{
  "symbol": "BTCUSD",
  "sl_pips": 75,
  "tp_pips": 150,
  "timeframe": "M1",
  "higher_timeframe": "M15",
  "strategy": "ma_crossover",
  "enable_trailing_stop": false,
  "max_daily_loss": 100,
  "max_daily_profit": 200,
  "backtest_period_days": 30
}
```

### 4. Run the Bot
```sh
python trading.py
```

## Configuration Options

| Setting | Description | Example |
|---------|-------------|---------|
| `symbol` | Trading instrument | `"BTCUSD"`, `"EURUSD"` |
| `timeframe` | Primary chart timeframe | `"M1"`, `"M5"`, `"H1"` |
| `higher_timeframe` | Trend confirmation timeframe | `"M15"`, `"H1"` |
| `strategy` | Trading strategy | `"ma_crossover"`, `"rsi"`, `"macd"`, `"bollinger"` |
| `sl_pips` | Stop loss in price units | `75` ($75 for crypto) |
| `tp_pips` | Take profit in price units | `150` ($150 for crypto) |
| `enable_trailing_stop` | Enable ATR trailing stop | `true`, `false` |
| `atr_period` | ATR calculation period | `14` |
| `atr_multiplier` | Trailing stop ATR multiplier | `1.5` |
| `min_atr` | Minimum ATR to trade (0=disabled) | `0` |
| `max_daily_loss` | Max daily loss before pause (USD) | `100` |
| `max_daily_profit` | Max daily profit before pause (USD) | `200` |
| `backtest_period_days` | Days of data for optimization | `30` |
| `news_block_minutes` | Minutes to block around news | `30` |

## How It Works

### Trading Flow
```
Every 60 seconds:
  1. Check market status
  2. Send hourly heartbeat (once per hour)
  3. Fetch price data (M1 bars)
  4. Check higher timeframe trend (M15)
  5. Apply ATR volatility filter
  6. Generate signal from strategy
  7. Execute trade if signal found
  8. Manage trailing stops
  9. Log to file (Telegram only for trades)
```

### Daily Optimization (8:00 AM ET)
```
  1. Run backtest.py
  2. Optimize all 4 strategies
  3. Rank by performance
  4. Auto-select best strategy
  5. Update config.json
  6. Reload parameters
  7. Send Telegram notification
```

## Telegram Notifications

**What sends to Telegram:**
- BUY/SELL orders placed
- Positions closed with P/L
- Trailing stop updates
- Hourly heartbeat with balance
- Daily strategy optimization results
- Max loss/profit alerts

**What does NOT send to Telegram:**
- "No trade signal" messages
- Filter messages (ATR, trend)
- Error messages
- Market closed messages

### Example Notifications
```
[HEARTBEAT] Bot running. Balance: $50000.10 | Strategy: ma_crossover | BTCUSD

[NOTIFY] BUY order placed, ticket: 224067017, price: 90673.81
[BALANCE] Account balance after BUY: 50150.00

[AUTOMATION] Best strategy selected: ma_crossover
[AUTOMATION] Parameters: {'short_ma': 80, 'long_ma': 180, 'total_return': 0.102}
```

## Files

| File | Description |
|------|-------------|
| `trading.py` | Main trading bot with all logic |
| `backtest.py` | Strategy optimization engine |
| `config.json` | All trading parameters |
| `strategy_params.json` | Auto-generated best parameters |
| `mt5_auth.json` | Credentials (DO NOT COMMIT) |
| `trade_notifications.log` | All events log |
| `trade_log.csv` | Closed trades history |
| `check_market.py` | Diagnostic tool |
| `check_broker_limits.py` | Symbol info checker |
| `test_trading.py` | Order test utility |

## Supported Brokers

Tested with:
- **Pepperstone** (recommended for crypto CFDs)
- **IC Markets**
- **MetaQuotes Demo**

The bot auto-detects the correct order filling mode (FOK/IOC) for each broker.

## Crypto vs Forex

| Setting | Forex (EURUSD) | Crypto (BTCUSD) |
|---------|----------------|-----------------|
| `symbol` | `"EURUSD"` | `"BTCUSD"` |
| `sl_pips` | `0.003` (30 pips) | `75` ($75) |
| `tp_pips` | `0.006` (60 pips) | `150` ($150) |
| `timeframe` | `"M5"` | `"M1"` (scalping) |
| Market hours | Mon-Fri | 24/7 |
| `backtest_period_days` | `180` | `30` |

## Security

- Never commit `mt5_auth.json`
- Always test with demo account first
- Enable AutoTrading in MT5 terminal
- Monitor Telegram for critical alerts

## Troubleshooting

### "AutoTrading disabled"
- Click AutoTrading button in MT5 toolbar (should be green)
- Check Tools > Options > Expert Advisors

### "Unsupported filling mode"
- Bot auto-detects correct mode
- If issues persist, check broker symbol settings

### No historical data
- Reduce `backtest_period_days` for M1 timeframe
- M1 data limited to ~30 days on most brokers

---

**Ready to trade!** Run `python trading.py` and let the bot find profitable opportunities.
