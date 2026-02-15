"""
Trade Journal Auto-Analysis
Reads today's trades + bot logs and uses Claude AI to generate deep insights.
Run daily or on-demand: python trade_journal.py [YYYY-MM-DD] [symbol]
"""
import os
import sys
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Config
# Try real API key first (OAuth tokens don't work for direct API calls)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY_REAL")
if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("sk-ant-oat"):
    # Read from polymarket .env which has the real key
    _poly_env = Path(__file__).parent.parent / "polymarket-bot" / ".env"
    if _poly_env.exists():
        for line in _poly_env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY=") and "api03" in line:
                ANTHROPIC_API_KEY = line.split("=", 1)[1].strip()
                break
if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("sk-ant-oat"):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SYMBOLS = ["btcusd", "xauusd", "eurusd"]
BOT_DIR = Path(__file__).parent


def get_trades_for_date(symbol: str, date_str: str) -> list[dict]:
    """Read trades from CSV for a specific date."""
    csv_path = BOT_DIR / symbol / "trade_log.csv"
    if not csv_path.exists():
        return []
    
    trades = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("close_time"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            close_time = parts[0]
            if date_str not in close_time:
                continue
            trades.append({
                "close_time": close_time,
                "direction": parts[1],
                "entry_price": float(parts[2]),
                "exit_price": float(parts[3]),
                "profit": float(parts[4]),
            })
    return trades


def get_log_snippets(symbol: str, date_str: str, max_lines: int = 200) -> str:
    """Extract relevant log lines for the date."""
    log_path = BOT_DIR / symbol / "trade_notifications.log"
    if not log_path.exists():
        return ""
    
    relevant = []
    keywords = [
        "[TRADE", "[ML]", "[TREND", "[FILTER", "[PARTIAL",
        "[BREAKEVEN", "[SMART EXIT", "[TRAILING", "[COOLDOWN",
        "[CRASH", "[VOLATILITY", "[SESSION", "signal blocked",
        "Spread too wide", "No trade signal", "[STARTUP]"
    ]
    
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if date_str not in line:
                continue
            # Only keep interesting lines
            if any(kw in line for kw in keywords):
                relevant.append(line.strip())
    
    # Limit size
    if len(relevant) > max_lines:
        # Keep first 50, last 50, and sample middle
        relevant = relevant[:80] + ["... (truncated) ..."] + relevant[-80:]
    
    return "\n".join(relevant)


def get_signal_stats(log_text: str) -> dict:
    """Parse log to count signals, blocks, and filters."""
    stats = {
        "total_signals": 0,
        "buy_signals": 0,
        "sell_signals": 0,
        "trend_blocked": 0,
        "spread_blocked": 0,
        "cooldown_blocked": 0,
        "volatility_blocked": 0,
        "no_signal": 0,
    }
    
    for line in log_text.split("\n"):
        if "[ML]" in line and ("BUY" in line or "SELL" in line):
            stats["total_signals"] += 1
            if "BUY" in line.split("[ML]")[1][:20]:
                stats["buy_signals"] += 1
            else:
                stats["sell_signals"] += 1
        if "signal blocked" in line.lower():
            stats["trend_blocked"] += 1
        if "Spread too wide" in line:
            stats["spread_blocked"] += 1
        if "[COOLDOWN]" in line:
            stats["cooldown_blocked"] += 1
        if "[VOLATILITY]" in line:
            stats["volatility_blocked"] += 1
        if "No trade signal" in line:
            stats["no_signal"] += 1
    
    return stats


def analyze_with_ai(symbol: str, date_str: str, trades: list, log_text: str, signal_stats: dict) -> str:
    """Send data to Claude for deep analysis."""
    import httpx
    
    if not ANTHROPIC_API_KEY:
        return "ERROR: No Anthropic API key found. Set ANTHROPIC_API_KEY_REAL in .env"
    
    # Build trade summary
    if trades:
        total_pnl = sum(t["profit"] for t in trades)
        wins = [t for t in trades if t["profit"] > 0]
        losses = [t for t in trades if t["profit"] <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        best = max(trades, key=lambda t: t["profit"])
        worst = min(trades, key=lambda t: t["profit"])
        
        trade_summary = f"""
TRADES ({len(trades)} total):
- P/L: ${total_pnl:.2f}
- Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {win_rate:.1f}%
- Best: {best['direction']} ${best['profit']:.2f} (entry {best['entry_price']:.2f} -> {best['exit_price']:.2f})
- Worst: {worst['direction']} ${worst['profit']:.2f} (entry {worst['entry_price']:.2f} -> {worst['exit_price']:.2f})

All trades:
""" + "\n".join(
            f"  {t['close_time'][:19]} | {t['direction']:4} | {t['entry_price']:.2f} -> {t['exit_price']:.2f} | ${t['profit']:+.2f}"
            for t in trades
        )
    else:
        trade_summary = "NO TRADES TODAY"
    
    stats_text = "\n".join(f"  {k}: {v}" for k, v in signal_stats.items())
    
    prompt = f"""You are a trading analyst reviewing an automated {symbol.upper()} trading bot's performance for {date_str}.

The bot uses:
- Ensemble ML (Random Forest + XGBoost + LightGBM) for entry signals
- EMA trend filter (blocks trades against the trend)
- ATR-based volatility filter
- Spread filter
- Adaptive cooldown after losses
- Partial profit taking at 1R (close 50%, move SL to breakeven)
- Smart exit (close stagnant trades after 30min)

{trade_summary}

SIGNAL STATISTICS:
{stats_text}

BOT LOGS (key events):
{log_text[:3000]}

Write a concise but insightful analysis covering:
1. **Performance Summary** - How did the bot do? Green or red day?
2. **Key Observations** - What patterns do you see? Timing, direction bias, filter effectiveness?
3. **Filter Analysis** - How many signals were blocked and was that good or bad? Did blocked signals would have been winners or losers?
4. **Risk Management** - Any concerning patterns? Position sizing, drawdown?
5. **Actionable Insights** - What should be adjusted? Any clear improvements?

Keep it to 200-300 words. Be direct, skip fluff. Use ASCII only (no Unicode arrows or special characters)."""

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    
    if response.status_code != 200:
        return f"API Error {response.status_code}: {response.text[:200]}"
    
    data = response.json()
    return data["content"][0]["text"]


def main():
    # Parse args
    target_date = None
    target_symbol = None
    
    for arg in sys.argv[1:]:
        if re.match(r"\d{4}-\d{2}-\d{2}", arg):
            target_date = arg
        elif arg.lower() in SYMBOLS:
            target_symbol = arg.lower()
    
    if not target_date:
        # Default to today UTC
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    symbols = [target_symbol] if target_symbol else SYMBOLS
    
    print(f"=== Trade Journal Analysis: {target_date} ===\n")
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  {symbol.upper()} - {target_date}")
        print(f"{'='*60}")
        
        trades = get_trades_for_date(symbol, target_date)
        log_text = get_log_snippets(symbol, target_date)
        signal_stats = get_signal_stats(log_text)
        
        if not trades and signal_stats["total_signals"] == 0:
            print(f"  No trades or signals for {symbol.upper()} on {target_date}")
            continue
        
        print(f"\n  Trades: {len(trades)} | Signals: {signal_stats['total_signals']} | Blocked: {signal_stats['trend_blocked']}")
        
        if trades:
            total = sum(t["profit"] for t in trades)
            print(f"  P/L: ${total:+.2f}")
        
        print(f"\n  Generating AI analysis...\n")
        
        analysis = analyze_with_ai(symbol, target_date, trades, log_text, signal_stats)
        print(analysis)
        print()
    
    print("\n=== End of Journal ===")


if __name__ == "__main__":
    main()
