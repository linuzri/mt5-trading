"""
Trade Journal Auto-Analysis
Reads today's trades + bot logs and uses Claude AI to generate deep insights.
Saves analysis to daily_analysis.json (dashboard) and Supabase.

Usage:
    python trade_journal.py [YYYY-MM-DD] [symbol]
    python trade_journal.py --save          # Analyze all bots + save to dashboard JSON + Supabase
    python trade_journal.py --save --date 2026-02-15
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
ANTHROPIC_API_KEY = None
SYMBOLS = ["btcusd", "xauusd", "eurusd"]
BOT_DIR = Path(__file__).parent
JSON_PATH = BOT_DIR / "vercel-dashboard" / "data" / "daily_analysis.json"
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://cxpablqwnwvacuvhcjen.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def _get_api_key():
    global ANTHROPIC_API_KEY
    if ANTHROPIC_API_KEY:
        return ANTHROPIC_API_KEY
    # Try real API key first (OAuth tokens don't work for direct API calls)
    key = os.getenv("ANTHROPIC_API_KEY_REAL")
    if not key or key.startswith("sk-ant-oat"):
        # Read from polymarket .env which has the real key
        _poly_env = Path(__file__).parent.parent / "polymarket-bot" / ".env"
        if _poly_env.exists():
            for line in _poly_env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY=") and "api03" in line:
                    key = line.split("=", 1)[1].strip()
                    break
    if not key or key.startswith("sk-ant-oat"):
        key = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_API_KEY = key
    return key


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
            if any(kw in line for kw in keywords):
                relevant.append(line.strip())

    if len(relevant) > max_lines:
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

    key = _get_api_key()
    if not key:
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
3. **Filter Analysis** - How many signals were blocked and was that good or bad?
4. **Risk Management** - Any concerning patterns? Position sizing, drawdown?
5. **Actionable Insights** - What should be adjusted? Any clear improvements?

Keep it to 200-300 words. Be direct, skip fluff. Use ASCII only (no Unicode arrows or special characters)."""

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
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


PORTFOLIO_PATH = Path(r"C:\Users\Nazri Hussain\projects\polymarket-bot\portfolio_state.json")
ARB_LOG_PATH = Path(r"C:\Users\Nazri Hussain\.pm2\logs\polymarket-arb-out.log")


def get_polymarket_summary() -> str:
    """Generate Polymarket portfolio summary from portfolio_state.json and PM2 logs."""
    if not PORTFOLIO_PATH.exists():
        return ""

    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception:
        return ""

    positions = data.get("positions", {})
    resolved = data.get("resolved", [])

    total_invested = sum(p.get("cost_basis", 0) for p in positions.values())
    current_value = sum(p.get("shares", 0) * p.get("current_price", 0) for p in positions.values())
    unrealized_pnl = current_value - total_invested
    realized_pnl = sum(r.get("realized_pnl", 0) for r in resolved)
    initial_deposit = 100.27

    # Parse sniper stats from PM2 arb logs (last 200 lines)
    sniper_trades = 0
    sniper_committed = 0
    sniper_limit = 0
    sniped_markets = 0
    arb_trades = 0
    tp_sells = 0

    if ARB_LOG_PATH.exists():
        try:
            with open(ARB_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-200:]
            for line in lines:
                # Sniper summary line: "Sniper: X trades placed ($Y committed / $Z limit) | N candidates found"
                m = re.search(r"Sniper: (\d+) trades placed \(\$(\d+) committed / \$(\d+) limit\)", line)
                if m:
                    sniper_trades = int(m.group(1))
                    sniper_committed = int(m.group(2))
                    sniper_limit = int(m.group(3))
                # Sniped markets count
                m = re.search(r"(\d+) candidates found", line)
                if m:
                    sniped_markets = int(m.group(1))
        except Exception:
            pass

    lines = [
        "### POLYMARKET",
        f"**Portfolio Summary**",
        f"- Open positions: {len(positions)}",
        f"- Total invested: ${total_invested:.2f}",
        f"- Current value: ${current_value:.2f}",
        f"- Unrealized P/L: ${unrealized_pnl:+.2f}",
        f"- Resolved: {len(resolved)}",
        f"- Realized P/L: ${realized_pnl:+.2f}",
        f"- Initial deposit: ${initial_deposit:.2f}",
        f"- Wallet balance: ${initial_deposit + realized_pnl - total_invested + current_value:.2f}",
        f"",
        f"**Sniper stats (this session):**",
        f"- Trades placed: {sniper_trades}",
        f"- Committed: ${sniper_committed} / ${sniper_limit} limit",
        f"- Candidates found: {sniped_markets}",
    ]

    # List open positions
    if positions:
        lines.append("")
        lines.append("**Open Positions:**")
        for name, p in positions.items():
            cost = p.get("cost_basis", 0)
            shares = p.get("shares", 0)
            price = p.get("current_price", 0)
            value = shares * price
            pnl = value - cost
            side = p.get("side", "?")
            lines.append(f"- {name[:50]} | {side} | cost ${cost:.2f} | value ${value:.2f} | P/L ${pnl:+.2f}")

    return "\n".join(lines)


def analyze_all_bots(date_str: str) -> dict:
    """Analyze all bots for a date, return structured result."""
    results = {}
    all_analyses = []

    for symbol in SYMBOLS:
        trades = get_trades_for_date(symbol, date_str)
        log_text = get_log_snippets(symbol, date_str)
        signal_stats = get_signal_stats(log_text)

        total_pnl = sum(t["profit"] for t in trades)
        wins = len([t for t in trades if t["profit"] > 0])
        losses = len([t for t in trades if t["profit"] <= 0])

        results[symbol] = {
            "trades": len(trades),
            "pnl": total_pnl,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(trades) * 100) if trades else 0,
            "signal_stats": signal_stats,
        }

        # Only call AI if there were trades or signals
        if trades or signal_stats["total_signals"] > 0:
            print(f"  Analyzing {symbol.upper()}...")
            analysis = analyze_with_ai(symbol, date_str, trades, log_text, signal_stats)
            results[symbol]["analysis"] = analysis
            header = f"### {symbol.upper()} -- ${total_pnl:+.2f}, {len(trades)} trades"
            if trades:
                header += f", {wins}W/{losses}L ({results[symbol]['win_rate']:.0f}% WR)"
            all_analyses.append(f"{header}\n{analysis}")
        else:
            results[symbol]["analysis"] = f"No trades or signals for {symbol.upper()} (market likely closed)."
            all_analyses.append(f"### {symbol.upper()}\nNo activity (market closed).")

    # Add Polymarket portfolio summary
    poly_summary = get_polymarket_summary()
    if poly_summary:
        all_analyses.append(poly_summary)

    # Combine into single summary
    total_pnl = sum(r["pnl"] for r in results.values())
    total_trades = sum(r["trades"] for r in results.values())
    combined = "\n\n".join(all_analyses)

    results["_combined"] = {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "summary": combined,
    }

    return results


def get_balance(date_str: str) -> float:
    """Get account balance from heartbeat logs."""
    for symbol in SYMBOLS:
        log_path = BOT_DIR / symbol / "trade_notifications.log"
        if not log_path.exists():
            continue
        balance = None
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if date_str in line and "[HEARTBEAT]" in line and "Balance:" in line:
                    m = re.search(r"Balance: \$([0-9,.]+)", line)
                    if m:
                        balance = float(m.group(1).replace(",", ""))
        if balance:
            return balance
    return 0.0


def save_to_json(date_str: str, results: dict, balance: float):
    """Save analysis to dashboard JSON file."""
    data = []
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Remove existing entry for this date
    data = [d for d in data if d.get("date") != date_str]

    entry = {
        "date": date_str,
        "summary": results["_combined"]["summary"],
        "balance": balance,
        "total_pnl": results["_combined"]["total_pnl"],
        "total_trades": results["_combined"]["total_trades"],
        "btcusd_pnl": results.get("btcusd", {}).get("pnl", 0),
        "xauusd_pnl": results.get("xauusd", {}).get("pnl", 0),
        "eurusd_pnl": results.get("eurusd", {}).get("pnl", 0),
    }
    data.append(entry)
    data.sort(key=lambda x: x["date"])

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {JSON_PATH}")


def save_to_supabase(date_str: str, results: dict, balance: float):
    """Save analysis to Supabase daily_analysis table."""
    if not SUPABASE_KEY:
        print("  Supabase key not found, skipping Supabase save.")
        return

    import httpx

    entry = {
        "date": date_str,
        "summary": results["_combined"]["summary"],
        "balance": balance,
        "total_pnl": results["_combined"]["total_pnl"],
        "total_trades": results["_combined"]["total_trades"],
        "btcusd_pnl": results.get("btcusd", {}).get("pnl", 0),
        "xauusd_pnl": results.get("xauusd", {}).get("pnl", 0),
        "eurusd_pnl": results.get("eurusd", {}).get("pnl", 0),
    }

    # Upsert (insert or update on conflict by date)
    resp = httpx.post(
        f"{SUPABASE_URL}/rest/v1/daily_analysis?on_conflict=date",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        json=entry,
        timeout=15,
    )
    if resp.status_code in (200, 201):
        print(f"  Saved to Supabase daily_analysis")
    else:
        print(f"  Supabase save failed ({resp.status_code}): {resp.text[:200]}")


def main():
    # Parse args
    target_date = None
    target_symbol = None
    save_mode = "--save" in sys.argv

    for arg in sys.argv[1:]:
        if re.match(r"\d{4}-\d{2}-\d{2}", arg):
            target_date = arg
        elif arg.lower() in SYMBOLS:
            target_symbol = arg.lower()
        elif arg.startswith("--date="):
            target_date = arg.split("=")[1]

    if not target_date:
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"=== Trade Journal Analysis: {target_date} ===\n")

    if save_mode or (not target_symbol):
        # Full analysis mode: all bots, save to dashboard + Supabase
        results = analyze_all_bots(target_date)
        balance = get_balance(target_date)

        # Print combined summary
        print(f"\n{results['_combined']['summary']}")

        if save_mode:
            print(f"\nSaving...")
            save_to_json(target_date, results, balance)
            save_to_supabase(target_date, results, balance)
            print(f"\nDone! Dashboard will update on next Vercel deploy/refresh.")
    else:
        # Single bot mode
        symbol = target_symbol
        print(f"{'='*60}")
        print(f"  {symbol.upper()} - {target_date}")
        print(f"{'='*60}")

        trades = get_trades_for_date(symbol, target_date)
        log_text = get_log_snippets(symbol, target_date)
        signal_stats = get_signal_stats(log_text)

        if not trades and signal_stats["total_signals"] == 0:
            print(f"  No trades or signals for {symbol.upper()} on {target_date}")
            return

        print(f"\n  Trades: {len(trades)} | Signals: {signal_stats['total_signals']} | Blocked: {signal_stats['trend_blocked']}")

        if trades:
            total = sum(t["profit"] for t in trades)
            print(f"  P/L: ${total:+.2f}")

        print(f"\n  Generating AI analysis...\n")
        analysis = analyze_with_ai(symbol, target_date, trades, log_text, signal_stats)
        print(analysis)

    print("\n=== End of Journal ===")


if __name__ == "__main__":
    main()
