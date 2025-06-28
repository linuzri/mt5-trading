# MT5-Trading Python Bot

This project is an automated MetaTrader 5 (MT5) trading bot in Python supporting multiple strategies (MA crossover, RSI, MACD, Bollinger Bands). It features automated weekly backtesting, parameter optimization, robust logging, and Telegram push notifications for all trade and system events.

## Features
- Automated trading on MT5 (MetaQuotes-Demo or your broker)
- Multiple strategies: MA crossover, RSI, MACD, Bollinger Bands
- All key settings configurable via `config.json`
- Automated weekly backtesting and parameter update
- Robust logging to `trade_notifications.log`
- Push notifications to Telegram (trades, errors, system events)
- Safe fallback if automation fails

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
5. **Run the bot:**
   ```sh
   python trading.py
   ```

## Notes
- `mt5_auth.json` is required and must not be committed to version control (see `.gitignore`).
- You must create your own Telegram bot and obtain your chat ID for notifications.
- All notifications are sent to Telegram, printed to the console, and logged in `trade_notifications.log`.

## Example Notification
```
[NOTIFY] BUY order placed, ticket: 123456789, price: 1.08500
[BALANCE] Account balance after BUY: 10005.23
[NOTIFY] Market is closed for EURUSD. Waiting...
```

---

For more details, see comments in `trading.py` and `config.json`.
