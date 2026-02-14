"""
Sync Polymarket sniper stats to Supabase for dashboard display.
Parses PM2 logs + portfolio_state.json to extract sniper performance data.
"""
import json
import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://cxpablqwnwvacuvhcjen.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service_role key
POLYMARKET_DIR = os.path.expanduser("~/projects/polymarket-bot")
PM2_LOG = os.path.expanduser("~/.pm2/logs/polymarket-arb-out.log")


def parse_pm2_logs():
    """Parse PM2 arb logs for sniper activity."""
    stats = {
        "orders_placed": 0,
        "candidates_found": 0,
        "cycles": 0,
        "exposure_limit_hits": 0,
        "balance_errors": 0,
        "markets_scanned": 0,
        "order_details": [],  # last N orders
        "first_seen": None,
        "last_seen": None,
    }

    if not os.path.exists(PM2_LOG):
        return stats

    with open(PM2_LOG, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Sniper order placed
            if "Sniper order placed:" in line:
                stats["orders_placed"] += 1
                # Extract timestamp
                ts_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
                if ts_match:
                    ts = ts_match.group(1)
                    if stats["first_seen"] is None:
                        stats["first_seen"] = ts
                    stats["last_seen"] = ts
                # Extract order details
                detail_match = re.search(r"Sniper order placed: (\w+) ([\d.]+) @ \$([\d.]+)", line)
                if detail_match:
                    stats["order_details"].append({
                        "side": detail_match.group(1),
                        "shares": float(detail_match.group(2)),
                        "price": float(detail_match.group(3)),
                        "timestamp": ts if ts_match else None
                    })

            # Cycle summary
            if "Sniper:" in line and "trades placed" in line:
                stats["cycles"] += 1
                m = re.search(r"(\d+) trades placed.*?(\d+) candidates found", line)
                if m:
                    stats["candidates_found"] += int(m.group(2))

            # Exposure limit
            if "exposure limit reached" in line.lower():
                stats["exposure_limit_hits"] += 1

            # Balance errors
            if "not enough balance" in line.lower():
                stats["balance_errors"] += 1

            # Markets scanned
            if "Scanned" in line and "markets" in line:
                m = re.search(r"Scanned (\d+) markets", line)
                if m:
                    stats["markets_scanned"] += int(m.group(1))

    # Keep only last 20 order details
    stats["order_details"] = stats["order_details"][-20:]
    return stats


def get_portfolio_data():
    """Read portfolio_state.json for positions and resolved trades."""
    path = os.path.join(POLYMARKET_DIR, "portfolio_state.json")
    if not os.path.exists(path):
        return {}

    with open(path) as f:
        data = json.load(f)

    positions = data.get("positions", {})
    resolved = data.get("resolved", [])

    # Separate sniper positions (price >= 0.95) from AI strategy positions
    sniper_positions = []
    strategy_positions = []
    for pid, pos in positions.items():
        entry = {
            "market": pos.get("market_question", "Unknown"),
            "side": pos.get("side", "?"),
            "shares": pos.get("shares", 0),
            "cost_basis": pos.get("cost_basis", 0),
            "avg_price": pos.get("avg_entry_price", 0),
            "current_price": pos.get("current_price", 0),
            "opened_at": pos.get("opened_at", ""),
        }
        if pos.get("avg_entry_price", 0) >= 0.90:
            sniper_positions.append(entry)
        else:
            strategy_positions.append(entry)

    # Separate resolved
    sniper_resolved = []
    strategy_resolved = []
    for r in resolved:
        entry = {
            "market": r.get("market_question", "Unknown"),
            "side": r.get("side", "?"),
            "shares": r.get("shares", 0),
            "cost_basis": r.get("cost_basis", 0),
            "avg_price": r.get("avg_entry_price", 0),
            "pnl": r.get("realized_pnl", 0),
            "outcome": r.get("outcome", "UNKNOWN"),
            "resolved_at": r.get("resolved_at", ""),
        }
        if r.get("avg_entry_price", 0) >= 0.90:
            sniper_resolved.append(entry)
        else:
            strategy_resolved.append(entry)

    return {
        "sniper_positions": sniper_positions,
        "strategy_positions": strategy_positions,
        "sniper_resolved": sniper_resolved,
        "strategy_resolved": strategy_resolved,
        "total_open": len(positions),
        "total_resolved": len(resolved),
    }


def get_strategy_trades():
    """Read strategy_trades.json for AI strategy trade history."""
    path = os.path.join(POLYMARKET_DIR, "strategy_trades.json")
    if not os.path.exists(path):
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("trades", [])


def build_sniper_dashboard_data():
    """Build the complete sniper dashboard dataset."""
    log_stats = parse_pm2_logs()
    portfolio = get_portfolio_data()
    strategy_trades = get_strategy_trades()

    # Calculate sniper P/L
    sniper_invested = sum(p["cost_basis"] for p in portfolio.get("sniper_positions", []))
    sniper_resolved_pnl = sum(r["pnl"] for r in portfolio.get("sniper_resolved", []))
    strategy_invested = sum(p["cost_basis"] for p in portfolio.get("strategy_positions", []))
    strategy_resolved_pnl = sum(r["pnl"] for r in portfolio.get("strategy_resolved", []))

    # Calculate fill rate (orders placed vs what we know filled)
    # For now, estimate based on positions that exist at sniper prices
    sniper_fills = len(portfolio.get("sniper_positions", [])) + len(portfolio.get("sniper_resolved", []))

    dashboard_data = {
        # Sniper stats
        "sniper_orders_placed": log_stats["orders_placed"],
        "sniper_fills": sniper_fills,
        "sniper_fill_rate": round(sniper_fills / max(log_stats["orders_placed"], 1) * 100, 1),
        "sniper_cycles": log_stats["cycles"],
        "sniper_candidates_per_cycle": round(log_stats["candidates_found"] / max(log_stats["cycles"], 1), 1),
        "sniper_exposure_limit_hits": log_stats["exposure_limit_hits"],
        "sniper_markets_scanned": log_stats["markets_scanned"],
        "sniper_invested": round(sniper_invested, 2),
        "sniper_resolved_pnl": round(sniper_resolved_pnl, 2),
        "sniper_positions": portfolio.get("sniper_positions", []),
        "sniper_resolved": portfolio.get("sniper_resolved", []),
        "sniper_recent_orders": log_stats["order_details"],
        "sniper_first_active": log_stats["first_seen"],
        "sniper_last_active": log_stats["last_seen"],

        # Strategy stats
        "strategy_trades": len(strategy_trades),
        "strategy_invested": round(strategy_invested, 2),
        "strategy_resolved_pnl": round(strategy_resolved_pnl, 2),
        "strategy_positions": portfolio.get("strategy_positions", []),
        "strategy_resolved": portfolio.get("strategy_resolved", []),

        # Overall
        "total_open_positions": portfolio.get("total_open", 0),
        "total_resolved": portfolio.get("total_resolved", 0),
        "bot_status": "running" if log_stats["last_seen"] else "unknown",

        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    return dashboard_data


def sync_to_supabase(data):
    """Upsert sniper stats to Supabase."""
    if not SUPABASE_KEY:
        print("[WARN] No SUPABASE_KEY, skipping sync")
        return

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    # Upsert to polymarket_portfolio table (reuse existing table)
    payload = {
        "id": "sniper_stats",
        "positions": json.dumps(data.get("sniper_positions", [])),
        "resolved": json.dumps(data.get("sniper_resolved", [])),
        "total_invested": data.get("sniper_invested", 0),
        "current_value": data.get("sniper_invested", 0),  # At $0.999, value ~ cost
        "unrealized_pnl": 0,
        "realized_pnl": data.get("sniper_resolved_pnl", 0),
        "initial_deposit": 100.27,
        "wallet_balance": 25.63,
        "updated_at": data["updated_at"],
    }

    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/polymarket_portfolio",
        headers=headers,
        json=payload
    )

    if resp.status_code in (200, 201):
        print(f"[OK] Synced sniper stats to Supabase")
    else:
        print(f"[ERROR] Supabase sync failed: {resp.status_code} {resp.text}")

    return data


def save_local(data):
    """Save sniper stats locally for dashboard to read."""
    out_path = os.path.join(os.path.dirname(__file__), "vercel-dashboard", "data", "sniper_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved sniper stats to {out_path}")


if __name__ == "__main__":
    print("Building sniper dashboard data...")
    data = build_sniper_dashboard_data()

    print(f"\nSniper Summary:")
    print(f"  Orders placed: {data['sniper_orders_placed']}")
    print(f"  Known fills: {data['sniper_fills']}")
    print(f"  Fill rate: {data['sniper_fill_rate']}%")
    print(f"  Scan cycles: {data['sniper_cycles']}")
    print(f"  Avg candidates/cycle: {data['sniper_candidates_per_cycle']}")
    print(f"  Invested: ${data['sniper_invested']}")
    print(f"  Resolved P/L: ${data['sniper_resolved_pnl']}")
    print(f"  Open sniper positions: {len(data['sniper_positions'])}")
    print(f"  Resolved sniper: {len(data['sniper_resolved'])}")
    print(f"  Strategy positions: {len(data['strategy_positions'])}")

    save_local(data)
    sync_to_supabase(data)
