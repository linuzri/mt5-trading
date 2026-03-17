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

    def evaluate(self, candles: pd.DataFrame, h4_candles: Optional[pd.DataFrame] = None, d1_candles: Optional[pd.DataFrame] = None) -> Optional[Signal]:
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

        # Enrichment calculations
        current_price = close.iloc[-1]
        
        # EMA gap calculations
        ema_gap = abs(fast_now - slow_now)
        ema_gap_pct = round((ema_gap / current_price) * 100, 4)
        
        # EMA trend (converging/diverging)
        prev_gap = abs(fast_prev - slow_prev)
        ema_trend = "converging" if ema_gap < prev_gap else "diverging"
        
        # Signal strength (normalized gap percentage)
        # Use a reasonable max gap of 2% for normalization
        max_gap_pct = 2.0
        signal_strength = min(ema_gap_pct / max_gap_pct, 1.0)
        
        # Candle body size relative to ATR
        last_candle = candles.iloc[-1]
        # Handle missing 'open' column gracefully
        if "open" in candles.columns:
            candle_body = abs(last_candle["close"] - last_candle["open"])
        else:
            # Fallback: estimate body size as small percentage of close price
            candle_body = last_candle["close"] * 0.001  # 0.1% as default body size
        candle_body_atr_ratio = round(candle_body / atr if atr > 0 else 0, 4)
        
        # H4 trend analysis
        h4_trend = "unavailable"
        h4_ema_fast = 0.0
        h4_ema_slow = 0.0
        
        if h4_candles is not None and len(h4_candles) >= slow_period:
            try:
                h4_close = h4_candles["close"]
                h4_fast = _ema(h4_close, fast_period)
                h4_slow = _ema(h4_close, slow_period)
                
                h4_ema_fast = round(h4_fast.iloc[-1], 2)
                h4_ema_slow = round(h4_slow.iloc[-1], 2)
                
                if h4_ema_fast > h4_ema_slow:
                    h4_trend = "above"
                elif h4_ema_fast < h4_ema_slow:
                    h4_trend = "below"
                else:
                    h4_trend = "neutral"
            except Exception:
                # Handle any errors gracefully
                h4_trend = "error"

        # D1 trend analysis
        d1_trend = "unavailable"
        d1_ema_fast = 0.0
        d1_ema_slow = 0.0
        
        if d1_candles is not None and len(d1_candles) >= slow_period:
            try:
                d1_close = d1_candles["close"]
                d1_fast = _ema(d1_close, fast_period)
                d1_slow = _ema(d1_close, slow_period)
                
                d1_ema_fast = round(d1_fast.iloc[-1], 2)
                d1_ema_slow = round(d1_slow.iloc[-1], 2)
                
                if d1_ema_fast > d1_ema_slow:
                    d1_trend = "above"
                elif d1_ema_fast < d1_ema_slow:
                    d1_trend = "below"
                else:
                    d1_trend = "neutral"
            except Exception:
                # Handle any errors gracefully
                d1_trend = "error"

        meta = {
            "ema_fast": round(fast_now, 2),
            "ema_slow": round(slow_now, 2),
            "ema_fast_prev": round(fast_prev, 2),
            "ema_slow_prev": round(slow_prev, 2),
            "atr": round(atr, 2),
            # Enrichment fields
            "ema_gap": round(ema_gap, 2),
            "ema_gap_pct": ema_gap_pct,
            "ema_trend": ema_trend,
            "signal_strength": round(signal_strength, 4),
            "candle_body_atr_ratio": candle_body_atr_ratio,
            "h4_trend": h4_trend,
            "h4_ema_fast": h4_ema_fast,
            "h4_ema_slow": h4_ema_slow,
            "d1_trend": d1_trend,
            "d1_ema_fast": d1_ema_fast,
            "d1_ema_slow": d1_ema_slow
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
