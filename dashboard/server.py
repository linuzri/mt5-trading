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
# Simple cache for balance history (expires after 30 seconds)
_balance_cache = {}
_cache_time = {}

BOTS = {
    'BTCUSD': {
        'name': 'BTCUSD Bot',
        'symbol': 'BTCUSD',
        'path': r'C:\Users\Nazri Hussain\projects\mt5-trading\\btcusd',
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

def get_balance_history(bot_key, days=30):
    """Get daily balance from logs to calculate REAL P/L (with caching)"""
    import time as time_module
    
    cache_key = bot_key
    now = time_module.time()
    
    # Return cached data if fresh (30 seconds)
    if cache_key in _balance_cache and cache_key in _cache_time:
        if now - _cache_time[cache_key] < 30:
            return _balance_cache[cache_key]
    
    bot = BOTS[bot_key]
    log_path = os.path.join(bot['path'], 'trade_notifications.log')
    
    daily_balances = {}  # date -> last balance of that day
    
    if not os.path.exists(log_path):
        return daily_balances
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '[BALANCE]' in line:
                    # Extract timestamp and balance
                    match = re.search(r'^(\d{4}-\d{2}-\d{2}).*\$([0-9,]+\.?\d*)', line)
                    if match:
                        date = match.group(1)
                        balance = float(match.group(2).replace(',', ''))
                        daily_balances[date] = balance
    except Exception as e:
        print(f"Error reading balance history: {e}")
    
    # Cache the result
    _balance_cache[cache_key] = daily_balances
    _cache_time[cache_key] = now
    
    return daily_balances

def get_daily_pnl(bot_key, days=30):
    """Calculate daily P/L from ACTUAL balance changes"""
    balance_history = get_balance_history(bot_key, days + 5)  # Get extra days for calculation
    
    if not balance_history:
        # Fallback to empty data
        result = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-1-i)).strftime('%Y-%m-%d')
            result.append({'date': date, 'pnl': 0})
        return result
    
    # Sort dates
    sorted_dates = sorted(balance_history.keys())
    
    # Calculate daily P/L from balance changes
    daily_pnl = {}
    prev_balance = None
    
    for date in sorted_dates:
        balance = balance_history[date]
        if prev_balance is not None:
            daily_pnl[date] = balance - prev_balance
        else:
            daily_pnl[date] = 0  # First day, no previous balance
        prev_balance = balance
    
    # Build result for requested days
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

def get_today_stats(bot_key):
    """Get today's trading stats for a bot"""
    trades = parse_trade_log(bot_key)
    today = datetime.now().strftime('%Y-%m-%d')
    
    today_trades = []
    for trade in trades:
        try:
            ts_str = trade['timestamp'].split('+')[0].split('.')[0]
            if ts_str.startswith(today):
                today_trades.append(trade)
        except:
            continue
    
    wins = sum(1 for t in today_trades if t['pnl'] > 0)
    losses = sum(1 for t in today_trades if t['pnl'] < 0)
    total_pnl = sum(t['pnl'] for t in today_trades)
    total_trades = len(today_trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'today_pnl': round(total_pnl, 2),
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1)
    }

@app.route('/api/account')
def api_account():
    """Get account balance (shared across all bots)"""
    # Get the most recent balance from any bot
    latest_balance = 0
    for bot_key in BOTS:
        balance = get_current_balance(bot_key)
        if balance > latest_balance:
            latest_balance = balance
    
    initial_fund = 50000.0
    total_pnl = latest_balance - initial_fund
    pnl_pct = (total_pnl / initial_fund) * 100
    
    return jsonify({
        'balance': round(latest_balance, 2),
        'initial': initial_fund,
        'pnl': round(total_pnl, 2),
        'pnl_pct': round(pnl_pct, 2)
    })

@app.route('/api/bots')
def api_bots():
    """Get all bot statuses and summary"""
    result = []
    
    for bot_key, bot in BOTS.items():
        # Get today's stats instead of total P/L
        today_stats = get_today_stats(bot_key)
        
        # Get recent performance for sparkline
        daily = get_daily_pnl(bot_key, 7)
        sparkline = [d['pnl'] for d in daily]
        
        result.append({
            'key': bot_key,
            'name': bot['name'],
            'symbol': bot['symbol'],
            'status': get_bot_status(bot_key),
            'today_pnl': today_stats['today_pnl'],
            'trades': today_stats['trades'],
            'wins': today_stats['wins'],
            'losses': today_stats['losses'],
            'win_rate': today_stats['win_rate'],
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
    """Get daily P/L for specified days with cumulative totals"""
    result = {}
    
    for bot_key in BOTS:
        daily = get_daily_pnl(bot_key, days)
        
        # Calculate cumulative P/L
        cumulative = 0
        for d in daily:
            cumulative += d['pnl']
            d['cumulative'] = round(cumulative, 2)
        
        result[bot_key] = daily
    
    # Also calculate total cumulative across all bots
    all_dates = result.get('BTCUSD', [])
    total_cumulative = []
    running_total = 0
    
    for i, d in enumerate(all_dates):
        day_total = sum(result[bot][i]['pnl'] for bot in result if i < len(result[bot]))
        running_total += day_total
        total_cumulative.append({
            'date': d['date'],
            'pnl': round(day_total, 2),
            'cumulative': round(running_total, 2)
        })
    
    result['_total'] = total_cumulative
    
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
