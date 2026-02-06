"""
Trading Bot Dashboard - Flask Backend Server
Reads real data from MT5 trading bots
Run: python server.py
Access: http://localhost:5000
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import re

app = Flask(__name__, static_folder='.')
CORS(app)

# Bot configurations
BOTS = {
    'BTCUSD': {
        'name': 'BTCUSD Bot',
        'symbol': 'BTCUSD',
        'path': r'C:\Users\Nazri Hussain\projects\mt5-trading\mt5-trading',
        'initial_fund': 50000.0,
        'color': '#f7931a'  # Bitcoin orange
    },
    'XAUUSD': {
        'name': 'XAUUSD Bot', 
        'symbol': 'XAUUSD',
        'path': r'C:\Users\Nazri Hussain\projects\mt5-trading\xauusd',
        'initial_fund': 50000.0,
        'color': '#ffd700'  # Gold
    },
    'EURUSD': {
        'name': 'EURUSD Bot',
        'symbol': 'EURUSD', 
        'path': r'C:\Users\Nazri Hussain\projects\mt5-trading\eurusd',
        'initial_fund': 50000.0,
        'color': '#0052b4'  # EU blue
    }
}

def parse_trade_log(bot_key):
    """Parse trade_log.csv for a bot"""
    bot = BOTS[bot_key]
    log_path = os.path.join(bot['path'], 'trade_log.csv')
    trades = []
    
    if not os.path.exists(log_path):
        return trades
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        timestamp = parts[0]
                        side = parts[1]
                        entry_price = float(parts[2])
                        exit_price = float(parts[3])
                        pnl = float(parts[4]) if len(parts) > 4 and parts[4] != 'N/A' else 0
                        
                        trades.append({
                            'timestamp': timestamp,
                            'side': side,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'bot': bot_key,
                            'symbol': bot['symbol']
                        })
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return trades

def parse_notification_log(bot_key, limit=50):
    """Parse trade_notifications.log for a bot"""
    bot = BOTS[bot_key]
    log_path = os.path.join(bot['path'], 'trade_notifications.log')
    logs = []
    
    if not os.path.exists(log_path):
        return logs
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Get last N lines
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                
                # Determine log level
                level = 'info'
                if '[ERROR]' in line or '[SHUTDOWN]' in line:
                    level = 'error'
                elif '[WARN]' in line:
                    level = 'warning'
                elif '[TRADE' in line or '[NOTIFY]' in line or '[CLOSE]' in line:
                    level = 'trade'
                elif '[BALANCE]' in line:
                    level = 'balance'
                
                logs.append({
                    'message': line,
                    'level': level,
                    'bot': bot_key
                })
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return logs

def get_current_balance(bot_key):
    """Get the most recent balance from logs"""
    bot = BOTS[bot_key]
    log_path = os.path.join(bot['path'], 'trade_notifications.log')
    
    if not os.path.exists(log_path):
        return bot['initial_fund']
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Search from end for balance
            for line in reversed(lines):
                if '[BALANCE]' in line:
                    match = re.search(r'\$([0-9,]+\.?\d*)', line)
                    if match:
                        return float(match.group(1).replace(',', ''))
    except Exception:
        pass
    
    return bot['initial_fund']

def get_bot_status(bot_key):
    """Check if bot is running by looking at recent log activity"""
    bot = BOTS[bot_key]
    log_path = os.path.join(bot['path'], 'trade_notifications.log')
    
    if not os.path.exists(log_path):
        return 'stopped'
    
    try:
        mtime = os.path.getmtime(log_path)
        age_seconds = (datetime.now().timestamp() - mtime)
        
        if age_seconds < 120:  # Updated in last 2 minutes
            return 'running'
        elif age_seconds < 600:  # Updated in last 10 minutes
            return 'paused'
        else:
            return 'stopped'
    except Exception:
        return 'stopped'

def get_daily_pnl(bot_key, days=30):
    """Calculate daily P/L from trades"""
    trades = parse_trade_log(bot_key)
    daily_pnl = defaultdict(float)
    
    cutoff = datetime.now() - timedelta(days=days)
    
    for trade in trades:
        try:
            # Parse timestamp
            ts_str = trade['timestamp'].split('+')[0].split('.')[0]
            ts = datetime.fromisoformat(ts_str)
            
            if ts >= cutoff:
                date_key = ts.strftime('%Y-%m-%d')
                daily_pnl[date_key] += trade['pnl']
        except Exception:
            continue
    
    # Fill in missing days with 0
    result = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days-1-i)).strftime('%Y-%m-%d')
        result.append({
            'date': date,
            'pnl': round(daily_pnl.get(date, 0), 2)
        })
    
    return result

def get_recent_trades(bot_key, limit=10):
    """Get recent trades"""
    trades = parse_trade_log(bot_key)
    return trades[-limit:][::-1]  # Reverse to get most recent first

def get_open_positions(bot_key):
    """Try to detect open positions from logs"""
    bot = BOTS[bot_key]
    log_path = os.path.join(bot['path'], 'trade_notifications.log')
    positions = []
    
    if not os.path.exists(log_path):
        return positions
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-200:]  # Check last 200 lines
            
            open_tickets = {}
            
            for line in lines:
                # Look for opened positions
                if '[NOTIFY]' in line and 'order placed' in line:
                    match = re.search(r'(BUY|SELL) order placed, ticket: (\d+), price: ([\d.]+)', line)
                    if match:
                        side, ticket, price = match.groups()
                        ts_match = re.match(r'([\d\-T:.]+)', line)
                        timestamp = ts_match.group(1) if ts_match else ''
                        open_tickets[ticket] = {
                            'ticket': ticket,
                            'side': side,
                            'entry_price': float(price),
                            'timestamp': timestamp,
                            'bot': bot_key,
                            'symbol': bot['symbol']
                        }
                
                # Remove closed positions
                if '[CLOSE]' in line:
                    match = re.search(r'position (\d+) closed', line)
                    if match:
                        ticket = match.group(1)
                        open_tickets.pop(ticket, None)
            
            positions = list(open_tickets.values())
    except Exception as e:
        print(f"Error getting positions: {e}")
    
    return positions

@app.route('/')
def index():
    return send_from_directory('.', 'trading-dashboard.html')

@app.route('/trading-dashboard.html')
def dashboard():
    return send_from_directory('.', 'trading-dashboard.html')

@app.route('/api/bots')
def api_bots():
    """Get all bot statuses and summary"""
    result = []
    
    for bot_key, bot in BOTS.items():
        current_balance = get_current_balance(bot_key)
        initial = bot['initial_fund']
        pnl = current_balance - initial
        pnl_pct = (pnl / initial) * 100 if initial > 0 else 0
        
        # Get recent performance for sparkline
        daily = get_daily_pnl(bot_key, 7)
        sparkline = [d['pnl'] for d in daily]
        
        result.append({
            'key': bot_key,
            'name': bot['name'],
            'symbol': bot['symbol'],
            'status': get_bot_status(bot_key),
            'initial_fund': initial,
            'current_fund': round(current_balance, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'color': bot['color'],
            'sparkline': sparkline
        })
    
    return jsonify(result)

@app.route('/api/daily-pnl')
def api_daily_pnl():
    """Get daily P/L for all bots"""
    result = {}
    
    for bot_key in BOTS:
        result[bot_key] = get_daily_pnl(bot_key, 30)
    
    return jsonify(result)

@app.route('/api/daily-pnl/<int:days>')
def api_daily_pnl_range(days):
    """Get daily P/L for specified days"""
    result = {}
    
    for bot_key in BOTS:
        result[bot_key] = get_daily_pnl(bot_key, days)
    
    return jsonify(result)

@app.route('/api/positions')
def api_positions():
    """Get all open positions"""
    positions = []
    
    for bot_key in BOTS:
        positions.extend(get_open_positions(bot_key))
    
    return jsonify(positions)

@app.route('/api/trades/<bot_key>')
def api_trades(bot_key):
    """Get recent trades for a bot"""
    if bot_key not in BOTS:
        return jsonify([])
    
    return jsonify(get_recent_trades(bot_key, 20))

@app.route('/api/logs/<bot_key>')
def api_logs(bot_key):
    """Get logs for a bot"""
    if bot_key not in BOTS:
        return jsonify([])
    
    return jsonify(parse_notification_log(bot_key, 50))

@app.route('/api/logs')
def api_all_logs():
    """Get logs for all bots"""
    result = {}
    
    for bot_key in BOTS:
        result[bot_key] = parse_notification_log(bot_key, 30)
    
    return jsonify(result)

if __name__ == '__main__':
    print("Trading Bot Dashboard Server")
    print("=" * 40)
    print("Access: http://localhost:5000")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000, debug=True)
