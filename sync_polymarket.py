"""Sync Polymarket portfolio_state.json to Supabase."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service role key
PORTFOLIO_PATH = Path(r"C:\Users\Nazri Hussain\projects\polymarket-bot\portfolio_state.json")


def main():
    if not PORTFOLIO_PATH.exists():
        print(f"Portfolio file not found: {PORTFOLIO_PATH}")
        return

    with open(PORTFOLIO_PATH, "r") as f:
        data = json.load(f)

    positions = data.get("positions", {})
    resolved = data.get("resolved", [])

    # Calculate totals
    total_invested = sum(p["cost_basis"] for p in positions.values())
    current_value = sum(p["shares"] * p["current_price"] for p in positions.values())
    unrealized_pnl = current_value - total_invested
    realized_pnl = sum(r["realized_pnl"] for r in resolved)

    row = {
        "id": 1,
        "positions": positions,
        "resolved": resolved,
        "total_invested": round(total_invested, 4),
        "current_value": round(current_value, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "realized_pnl": round(realized_pnl, 4),
        "updated_at": data.get("last_updated"),
    }

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = client.table("polymarket_portfolio").upsert(row).execute()
    print(f"Synced: invested=${row['total_invested']:.2f}, value=${row['current_value']:.2f}, "
          f"unrealized=${row['unrealized_pnl']:.2f}, realized=${row['realized_pnl']:.2f}")


if __name__ == "__main__":
    main()
