#!/usr/bin/env python3
"""
Daily Digest Generator for MT5 Trading Bots
Generates daily performance reports with P/L per bot and account balance changes.
"""

import os
import re
import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Bot configurations
BOTS = {
    "BTCUSD": {
        "path": r"C:\Users\Nazri Hussain\projects\mt5-trading\mt5-trading",
        "trade_log": "trade_log.csv",
        "notification_log": "trade_notifications.log"
    },
    "XAUUSD": {
        "path": r"C:\Users\Nazri Hussain\projects\mt5-trading\xauusd",
        "trade_log": "trade_log.csv",
        "notification_log": "trade_notifications.log"
    },
    "EURUSD": {
        "path": r"C:\Users\Nazri Hussain\projects\mt5-trading\eurusd",
        "trade_log": "trade_log.csv",
        "notification_log": "trade_notifications.log"
    }
}

# Output paths
DIGEST_FOLDER = r"C:\Users\Nazri Hussain\.openclaw\workspace\memory\trading"
BALANCE_HISTORY = os.path.join(DIGEST_FOLDER, "balance_history.json")


def ensure_folders():
    """Create necessary folders if they don't exist."""
    os.makedirs(DIGEST_FOLDER, exist_ok=True)


def parse_date(date_str):
    """Parse date string from various formats."""
    # Handle ISO format with timezone: 2026-02-02T15:31:07.271985+00:00
    if 'T' in date_str:
        date_part = date_str.split('T')[0]
        return datetime.strptime(date_part, '%Y-%m-%d').date()
    return datetime.strptime(date_str[:10], '%Y-%m-%d').date()


def get_trades_for_date(bot_name, bot_config, target_date):
    """Get all trades for a specific bot on a specific date."""
    trade_log_path = os.path.join(bot_config["path"], bot_config["trade_log"])
    trades = []
    
    if not os.path.exists(trade_log_path):
        return trades
    
    with open(trade_log_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                trade_date = parse_date(row[0])
                if trade_date == target_date:
                    trades.append({
                        'timestamp': row[0],
                        'direction': row[1],
                        'entry_price': float(row[2]) if row[2] else 0,
                        'exit_price': float(row[3]) if row[3] else 0,
                        'profit': float(row[4]) if row[4] not in ['N/A', ''] else None
                    })
            except (ValueError, IndexError):
                continue
    
    return trades


def get_balance_from_heartbeats(bot_config, target_date):
    """
    Extract first and last balance from heartbeat logs for a specific date.
    Returns (opening_balance, closing_balance) or (None, None) if not found.
    """
    log_path = os.path.join(bot_config["path"], bot_config["notification_log"])
    
    if not os.path.exists(log_path):
        return None, None
    
    # Regex to match heartbeat lines with balance
    heartbeat_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}.*\[HEARTBEAT\].*Balance: \$([0-9,.]+)'
    )
    
    balances_for_date = []
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = heartbeat_pattern.search(line)
            if match and match.group(1) == target_date_str:
                balance = float(match.group(2).replace(',', ''))
                balances_for_date.append(balance)
    
    if not balances_for_date:
        return None, None
    
    return balances_for_date[0], balances_for_date[-1]


def load_balance_history():
    """Load historical balance data."""
    if os.path.exists(BALANCE_HISTORY):
        with open(BALANCE_HISTORY, 'r') as f:
            return json.load(f)
    return {}


def save_balance_history(history):
    """Save balance history to JSON."""
    with open(BALANCE_HISTORY, 'w') as f:
        json.dump(history, f, indent=2)


def calculate_bot_stats(trades):
    """Calculate statistics for a list of trades."""
    if not trades:
        return {
            'total_trades': 0,
            'buys': 0,
            'sells': 0,
            'profit': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0
        }
    
    trades_with_profit = [t for t in trades if t['profit'] is not None]
    total_profit = sum(t['profit'] for t in trades_with_profit)
    wins = sum(1 for t in trades_with_profit if t['profit'] > 0)
    losses = sum(1 for t in trades_with_profit if t['profit'] < 0)
    
    return {
        'total_trades': len(trades),
        'buys': sum(1 for t in trades if t['direction'] == 'BUY'),
        'sells': sum(1 for t in trades if t['direction'] == 'SELL'),
        'profit': total_profit,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / len(trades_with_profit) * 100) if trades_with_profit else 0
    }


def generate_digest(target_date=None):
    """Generate daily digest for a specific date (defaults to today)."""
    ensure_folders()
    
    if target_date is None:
        target_date = datetime.now().date()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    date_str = target_date.strftime('%Y-%m-%d')
    
    # Collect data from all bots
    bot_reports = {}
    total_profit = 0
    total_trades = 0
    opening_balance = None
    closing_balance = None
    
    for bot_name, bot_config in BOTS.items():
        trades = get_trades_for_date(bot_name, bot_config, target_date)
        stats = calculate_bot_stats(trades)
        bot_reports[bot_name] = {
            'trades': trades,
            'stats': stats
        }
        total_profit += stats['profit']
        total_trades += stats['total_trades']
        
        # Get balance from any bot (they share the same account)
        if opening_balance is None:
            o_bal, c_bal = get_balance_from_heartbeats(bot_config, target_date)
            if o_bal is not None:
                opening_balance = o_bal
                closing_balance = c_bal
    
    # Load and update balance history
    history = load_balance_history()
    
    # Get previous day's closing balance for comparison
    prev_date = (target_date - timedelta(days=1)).strftime('%Y-%m-%d')
    prev_closing = history.get(prev_date, {}).get('closing_balance')
    
    # Use previous closing as today's opening if we don't have heartbeat data
    if opening_balance is None and prev_closing:
        opening_balance = prev_closing
    
    # Calculate actual balance change
    balance_change = 0
    if opening_balance and closing_balance:
        balance_change = closing_balance - opening_balance
    
    # Save today's data to history
    history[date_str] = {
        'opening_balance': opening_balance,
        'closing_balance': closing_balance,
        'total_trades': total_trades,
        'total_profit': total_profit,
        'by_symbol': {
            symbol: report['stats'] for symbol, report in bot_reports.items()
        }
    }
    save_balance_history(history)
    
    # Generate markdown digest
    open_str = f"${opening_balance:,.2f}" if opening_balance else "N/A"
    close_str = f"${closing_balance:,.2f}" if closing_balance else "N/A"
    change_str = f"${balance_change:+,.2f}" if balance_change else "N/A"
    
    digest = f"""# Daily Trading Digest - {date_str}

## 📊 Account Summary

| Metric | Value |
|--------|-------|
| Opening Balance | {open_str} |
| Closing Balance | {close_str} |
| Balance Change | {change_str} |
| Total P/L (Trades) | ${total_profit:+,.2f} |
| Total Trades | {total_trades} |

## 🤖 Bot Performance

"""
    
    for bot_name, report in bot_reports.items():
        stats = report['stats']
        emoji = "🟢" if stats['profit'] >= 0 else "🔴"
        
        digest += f"""### {bot_name} {emoji}

| Metric | Value |
|--------|-------|
| Trades | {stats['total_trades']} (Buy: {stats['buys']}, Sell: {stats['sells']}) |
| P/L | ${stats['profit']:+,.2f} |
| Win Rate | {stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L) |

"""
        
        # Add trade details if any
        if report['trades']:
            digest += "**Trade Details:**\n\n"
            digest += "| Time | Direction | Entry | Exit | P/L |\n"
            digest += "|------|-----------|-------|------|-----|\n"
            for trade in report['trades']:
                ts = trade['timestamp']
                time_part = ts.split('T')[1][:8] if 'T' in ts else ts[11:19] if len(ts) > 19 else 'N/A'
                pl_str = f"${trade['profit']:+,.2f}" if trade['profit'] is not None else "N/A"
                digest += f"| {time_part} | {trade['direction']} | {trade['entry_price']:.2f} | {trade['exit_price']:.2f} | {pl_str} |\n"
            digest += "\n"
    
    # Add weekly summary if it's end of week
    if target_date.weekday() == 4:  # Friday
        digest += generate_weekly_summary(target_date, history)
    
    # Save digest to file
    digest_file = os.path.join(DIGEST_FOLDER, f"{date_str}.md")
    with open(digest_file, 'w', encoding='utf-8') as f:
        f.write(digest)
    
    print(f"[OK] Digest saved to: {digest_file}")
    return digest, digest_file


def generate_weekly_summary(end_date, history):
    """Generate weekly summary section."""
    # Get last 7 days
    summary = "\n## 📈 Weekly Summary\n\n"
    
    week_profit = 0
    week_trades = 0
    week_opening = None
    week_closing = None
    
    for i in range(7):
        day = (end_date - timedelta(days=6-i))
        day_str = day.strftime('%Y-%m-%d')
        day_data = history.get(day_str, {})
        
        if day_data:
            week_profit += day_data.get('total_profit', 0)
            week_trades += day_data.get('total_trades', 0)
            
            if week_opening is None and day_data.get('opening_balance'):
                week_opening = day_data['opening_balance']
            if day_data.get('closing_balance'):
                week_closing = day_data['closing_balance']
    
    week_change = (week_closing - week_opening) if (week_opening and week_closing) else 0
    
    wo_str = f"${week_opening:,.2f}" if week_opening else "N/A"
    wc_str = f"${week_closing:,.2f}" if week_closing else "N/A"
    wch_str = f"${week_change:+,.2f}" if week_change else "N/A"
    
    summary += f"""| Week Stats | Value |
|------------|-------|
| Starting Balance | {wo_str} |
| Ending Balance | {wc_str} |
| Week Change | {wch_str} |
| Week P/L | ${week_profit:+,.2f} |
| Total Trades | {week_trades} |

"""
    return summary


def generate_telegram_summary(target_date=None):
    """Generate a compact summary for Telegram notification."""
    if target_date is None:
        target_date = datetime.now().date()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    date_str = target_date.strftime('%Y-%m-%d')
    
    # Collect data
    lines = [f"📊 *Daily Trading Report - {date_str}*\n"]
    total_profit = 0
    total_trades = 0
    opening_balance = None
    closing_balance = None
    
    for bot_name, bot_config in BOTS.items():
        trades = get_trades_for_date(bot_name, bot_config, target_date)
        stats = calculate_bot_stats(trades)
        
        if opening_balance is None:
            o_bal, c_bal = get_balance_from_heartbeats(bot_config, target_date)
            if o_bal:
                opening_balance, closing_balance = o_bal, c_bal
        
        emoji = "🟢" if stats['profit'] >= 0 else "🔴" if stats['profit'] < 0 else "⚪"
        lines.append(f"{emoji} *{bot_name}*: {stats['total_trades']} trades, ${stats['profit']:+.2f}")
        total_profit += stats['profit']
        total_trades += stats['total_trades']
    
    # Add totals
    lines.append("")
    if opening_balance and closing_balance:
        change = closing_balance - opening_balance
        lines.append(f"💰 *Balance*: ${opening_balance:,.2f} → ${closing_balance:,.2f} ({change:+.2f})")
    lines.append(f"📈 *Total P/L*: ${total_profit:+.2f} from {total_trades} trades")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    # Parse command line args
    target_date = None
    telegram_mode = False
    
    for arg in sys.argv[1:]:
        if arg == '--telegram':
            telegram_mode = True
        elif arg.startswith('--date='):
            target_date = arg.split('=')[1]
        elif re.match(r'\d{4}-\d{2}-\d{2}', arg):
            target_date = arg
    
    if telegram_mode:
        print(generate_telegram_summary(target_date))
    else:
        digest, filepath = generate_digest(target_date)
        print(f"\n{'='*50}")
        print(digest)
