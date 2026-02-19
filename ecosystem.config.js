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
        TELEGRAM_BOT_TOKEN: "7088277359:AAGU_s4uqXfOenQp9eFHCgB-NqV7aE5o9pw"
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
    }
  ]
}
