module.exports = {
  apps: [
    {
      name: "bot-btcusd-v2",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\btcusd",
      script: "main.py",
      interpreter: "python",
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
        TELEGRAM_BOT_TOKEN: process.env.TELEGRAM_BOT_TOKEN || "",
        TELEGRAM_CHAT_ID: "3588682"
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
    }
  ]
}
