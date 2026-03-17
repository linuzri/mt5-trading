"""Daily/weekly trade caps, cooldown, open position limits."""
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class Limits:
    def __init__(self, config: dict) -> None:
        self.config = config

    def check(self, state: dict, open_position_count: int) -> tuple:
        """
        Check all risk limits against current state.
        Returns (allowed: bool, reason: str).
        """
        now = datetime.now(timezone.utc)

        # Max open positions
        max_open = self.config.get("max_open_positions", 3)
        if open_position_count >= max_open:
            return False, f"max_open_positions ({open_position_count}/{max_open})"

        # Daily trade cap
        daily = state.get("daily_trade_count", 0)
        max_daily = self.config.get("max_daily_trades", 50)
        if daily >= max_daily:
            return False, f"max_daily_trades ({daily}/{max_daily})"

        # Weekly trade cap
        weekly = state.get("weekly_trade_count", 0)
        max_weekly = self.config.get("max_weekly_trades", 200)
        if weekly >= max_weekly:
            return False, f"max_weekly_trades ({weekly}/{max_weekly})"

        # Cooldown between trades
        last_trade_ts = state.get("last_trade_ts")
        cooldown = self.config.get("cooldown_seconds", 300)
        if last_trade_ts:
            elapsed = (now - datetime.fromisoformat(last_trade_ts)).total_seconds()
            if elapsed < cooldown:
                remaining = int(cooldown - elapsed)
                return False, f"cooldown_active (remaining={remaining}s)"

        return True, "ok"
