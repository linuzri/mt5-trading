# MT5-Trading Python Bot

A robust, fully automated MetaTrader 5 (MT5) trading bot in Python supporting multiple strategies, advanced risk management, and comprehensive notifications/logging.

## Features
- **Automated trading on MT5** (MetaQuotes-Demo or your broker)
- **Multiple strategies:** MA crossover, RSI, MACD, Bollinger Bands
- **All key settings configurable** via `config.json`
- **Automated weekly backtesting and parameter optimization** (runs every Monday 8:00–12:00 US Eastern)
- **Multi-timeframe confirmation:** Only trades when lower and higher timeframes agree
- **ATR volatility filter:** Trades only when volatility is above a configurable threshold
- **News event avoidance:** Placeholder for real news API integration (ready for extension)
- **Robust logging:** All actions, errors, and events logged to `trade_notifications.log`
- **Trade logging:** All closed trades logged to `trade_log.csv` and in-memory for summaries
- **Telegram notifications:** All trades, errors, summaries, and critical alerts sent to Telegram
- **Heartbeat notification:** Bot confirms it is running when market is closed
- **Critical alerts:** Max daily loss/profit triggers Telegram alert and trading pause
- **Trailing stop loss:** ATR-based trailing stop management (configurable)
- **Suppression of repeated filter messages** to avoid notification spam
- **Safe fallback:** If automation or backtesting fails, bot continues with previous parameters
- **Easy setup:** Credentials and sensitive info separated in `mt5_auth.json`

## Requirements
- Python 3.8+
- MetaTrader5 (`pip install MetaTrader5`)
- pandas (`pip install pandas`)
- numpy (`pip install numpy`)
- requests (`pip install requests`)
- pytz or zoneinfo (for timezone handling)

## Setup
1. **Clone this repository**
2. **Install dependencies:**
   ```sh
   pip install MetaTrader5 pandas numpy requests pytz
   ```
3. **Create your `mt5_auth.json` file in the project directory:**
   (Do NOT commit your real credentials)
   ```json
   {
     "login": 123456789,
     "password": "YOUR_MT5_PASSWORD",
     "server": "YOUR_MT5_SERVER",
     "telegram": {
       "bot_name": "@your_telegram_bot",
       "api_token": "YOUR_TELEGRAM_BOT_TOKEN",
       "chat_id": 123456789
     }
   }
   ```
4. **Edit `config.json` to set your trading parameters.**
   - Example of important config options:
     - `strategy`: "ma_crossover", "rsi", "macd", "bollinger"
     - `timeframe`: "M5", "M15", etc.
     - `higher_timeframe`: "H1", etc. (for trend confirmation)
     - `sl_pips`, `tp_pips`: Stop loss/take profit in pips
     - `atr_period`, `atr_multiplier`, `min_atr`: ATR filter/trailing stop settings
     - `max_daily_loss`, `max_daily_profit`: Daily risk limits (USD)
     - `news_block_minutes`: Minutes before/after news to block trading
     - `enable_trailing_stop`: true/false
5. **Run the bot:**
   ```sh
   python trading.py
   ```

## How It Works
- The bot runs in a continuous loop, checking for trade signals and managing open trades.
- Every Monday after 8:00 AM US Eastern, it runs `backtest.py` to optimize parameters and reloads them automatically.
- Trades are only placed if all filters (multi-timeframe, ATR, news) pass.
- All actions, trades, and errors are logged and sent to Telegram.
- Daily and weekly trade summaries are sent to Telegram.
- If max daily loss/profit is reached, the bot pauses trading and sends an alert.
- Heartbeat notifications are sent when the market is closed.

## Example Notification
```
[NOTIFY] BUY order placed, ticket: 123456789, price: 1.08500
[BALANCE] Account balance after BUY: 10005.23
[HEARTBEAT] Market is closed for EURUSD. Bot is running. UTC: 2025-07-01T00:00:00+00:00, ET: 2025-06-30T20:00:00-04:00
[SUMMARY] Daily Trade Summary: ...
[ALERT] Max daily loss reached: -100.00 USD. Trading paused.
```

## Files
- `trading.py` — Main trading bot
- `backtest.py` — Backtesting and parameter optimization
- `config.json` — All key parameters and settings
- `strategy_params.json` — Stores best parameters/results for each strategy
- `trade_notifications.log` — Logs all trade/automation events
- `trade_log.csv` — Logs all closed trades for summary/alerting
- `mt5_auth.json` — MT5 and Telegram credentials (never commit this file)

## Extending & Customization
- To add a real news filter, update the `is_high_impact_news_near` function in `trading.py` to call a news API.
- To add more strategies, extend the strategy logic in `trading.py` and update `backtest.py` accordingly.
- For multi-symbol trading, refactor the main loop to iterate over a list of symbols.

## Security & Best Practices
- Never commit your real `mt5_auth.json`.
- Always test with a demo account before using real funds.
- Monitor Telegram notifications for critical alerts and summaries.

---
For more details, see comments in `trading.py` and `config.json`.
