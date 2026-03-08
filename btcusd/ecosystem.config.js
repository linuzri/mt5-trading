module.exports = {
  apps: [
    {
      name: "bot-btcusd",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\btcusd",
      script: "trading.py",
      interpreter: "python",
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "bot-xauusd",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\xauusd",
      script: "trading.py",
      interpreter: "python",
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "bot-eurusd",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\eurusd",
      script: "trading.py",
      interpreter: "python",
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "mt5-watchdog",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading",
      script: "mt5_watchdog.js",
      autorestart: true,
      watch: false,
      env: {
        TELEGRAM_BOT_TOKEN: process.env.TELEGRAM_BOT_TOKEN || ""
      }
    },
    {
      name: "bot-btcusd-live",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\btcusd-live",
      script: "trading.py",
      interpreter: "python",
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },

    // ── AutoResearch loop ───────────────────────────────────────────────────
    {
      name: "autotrader",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\btcusd",
      script: "python_run.cmd",
      args: "autotrader.py --hours 168 --delay 120",
      autorestart: true,
      max_restarts: 5,
      restart_delay: 30000,
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1",
        TELEGRAM_BOT_TOKEN_AUTORESEARCH: "8603433804:AAFlKOclWaiuj8FtB07tv3G13wIYaIbsWz4",
        TELEGRAM_CHAT_ID_AUTORESEARCH: "3588682",
        ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY || ""
      }
    }
  ]
}
