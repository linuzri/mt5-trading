"""Position sizing — % risk per trade."""
import logging
import math

log = logging.getLogger(__name__)


class Sizer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def calculate_lot(self, balance: float, sl_distance: float) -> float:
        """
        Calculate lot size so that a loss of sl_distance equals risk_per_trade % of balance.

        For BTCUSD (1 lot = 1 BTC), 1 pip = $1. So:
            risk_$  = lot * sl_distance
            lot     = risk_$ / sl_distance
        """
        if sl_distance <= 0 or balance <= 0:
            log.warning("Invalid sizing inputs: balance=%.2f sl_dist=%.2f", balance, sl_distance)
            return self.config.get("min_lot", 0.01)

        risk_pct = self.config.get("risk_per_trade", 0.5) / 100.0
        risk_dollars = balance * risk_pct
        raw_lot = risk_dollars / sl_distance

        min_lot = self.config.get("min_lot", 0.01)
        max_lot = self.config.get("max_lot", 1.0)

        # Round down to 2 decimal places (broker standard)
        lot = math.floor(raw_lot * 100) / 100
        lot = max(min_lot, min(max_lot, lot))

        log.debug("Sizing: balance=%.2f risk_pct=%.2f sl_dist=%.2f lot=%.2f",
                  balance, risk_pct * 100, sl_distance, lot)
        return lot
