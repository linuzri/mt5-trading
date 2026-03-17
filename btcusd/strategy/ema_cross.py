"""EMA Crossover strategy — dead simple cross and go."""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import Signal, Strategy

log = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


class EmaCrossStrategy(Strategy):
    @property
    def name(self) -> str:
        return "ema_cross"

    def evaluate(self, candles: pd.DataFrame) -> Optional[Signal]:
        fast_period = self.config.get("ema_fast", 10)
        slow_period = self.config.get("ema_slow", 50)
        atr_period = self.config.get("atr_period", 14)
        sl_mult = self.config.get("sl_atr_multiplier", 2.0)
        tp_mult = self.config.get("tp_atr_multiplier", 3.0)

        min_candles = slow_period + 2
        if candles is None or len(candles) < min_candles:
            log.debug("Not enough candles (%d < %d)", len(candles) if candles is not None else 0, min_candles)
            return None

        close = candles["close"]
        fast = _ema(close, fast_period)
        slow = _ema(close, slow_period)

        # Current and previous bar values
        fast_now = fast.iloc[-1]
        fast_prev = fast.iloc[-2]
        slow_now = slow.iloc[-1]
        slow_prev = slow.iloc[-2]

        atr = _atr(candles, atr_period)
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult

        meta = {
            "ema_fast": round(fast_now, 2),
            "ema_slow": round(slow_now, 2),
            "ema_fast_prev": round(fast_prev, 2),
            "ema_slow_prev": round(slow_prev, 2),
            "atr": round(atr, 2),
        }

        # Bullish crossover: fast crosses above slow
        if fast_prev <= slow_prev and fast_now > slow_now:
            log.info("EMA crossover UP: fast=%.2f slow=%.2f atr=%.2f", fast_now, slow_now, atr)
            return Signal(
                direction="buy",
                reason=f"EMA{fast_period} crossed above EMA{slow_period}",
                sl_distance=sl_dist,
                tp_distance=tp_dist,
                metadata=meta,
            )

        # Bearish crossover: fast crosses below slow
        if fast_prev >= slow_prev and fast_now < slow_now:
            log.info("EMA crossover DOWN: fast=%.2f slow=%.2f atr=%.2f", fast_now, slow_now, atr)
            return Signal(
                direction="sell",
                reason=f"EMA{fast_period} crossed below EMA{slow_period}",
                sl_distance=sl_dist,
                tp_distance=tp_dist,
                metadata=meta,
            )

        return None
