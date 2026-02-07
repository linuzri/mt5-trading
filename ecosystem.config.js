module.exports = {
  apps: [{
    name: 'mt5-trading-bot',
    script: 'start_bot.bat',
    cwd: 'C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\mt5-trading',
    interpreter: 'cmd',
    interpreter_args: '/c',
    watch: false,
    autorestart: true,
    max_restarts: 10,
    restart_delay: 5000,
    // Logging
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    error_file: 'C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\mt5-trading\\logs\\pm2-error.log',
    out_file: 'C:\\Users\\Nazri Hussain\\projects\\mt5-trading\\mt5-trading\\logs\\pm2-out.log',
    merge_logs: true
  }]
};
