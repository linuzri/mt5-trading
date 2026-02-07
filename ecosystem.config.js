module.exports = {
  apps: [
    {
      name: "bot-btcusd",
      cwd: "C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\btcusd",
      script: "start_bot.bat",
      interpreter: "cmd",
      interpreter_args: "/c",
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
      script: "start_bot.bat",
      interpreter: "cmd",
      interpreter_args: "/c",
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
      script: "start_bot.bat",
      interpreter: "cmd",
      interpreter_args: "/c",
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
}
