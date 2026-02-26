"""
Trend-following strategy for BTCUSD H1.

Direction is determined by higher-timeframe trend (H4/D1 EMA alignment).
ML model acts as quality filter, not direction predictor.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, UTC


class TrendStrategy:

    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol

    def get_trend_direction(self):
        """
        Determine trend from H4 EMA alignment.

        Returns: 'bullish', 'bearish', or 'neutral'

        Bullish: Price > EMA20 > EMA50 on H4
        Bearish: Price < EMA20 < EMA50 on H4
        Neutral: EMAs tangled (no trade)
        """
        rates = mt5.copy_rates_from(self.symbol, mt5.TIMEFRAME_H4, datetime.now(UTC), 100)
        if rates is None or len(rates) < 60:
            return 'neutral'

        df = pd.DataFrame(rates)
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        latest = df.iloc[-1]
        price = latest['close']
        ema20 = latest['ema20']
        ema50 = latest['ema50']

        if price > ema20 > ema50:
            return 'bullish'
        elif price < ema20 < ema50:
            return 'bearish'
        else:
            return 'neutral'

    def get_h1_entry_signal(self, trend_direction):
        """
        H1 pullback entry: wait for price to pull back to EMA20 on H1
        then enter in the trend direction.

        BUY entry: H4 bullish + H1 candle touches/crosses below EMA20 then closes above
        SELL entry: H4 bearish + H1 candle touches/crosses above EMA20 then closes below
        """
        rates = mt5.copy_rates_from(self.symbol, mt5.TIMEFRAME_H1, datetime.now(UTC), 50)
        if rates is None or len(rates) < 30:
            return None, "insufficient data"

        df = pd.DataFrame(rates)
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['atr'] = self._calc_atr(df, 14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = latest['close']
        ema20 = latest['ema20']
        low = latest['low']
        high = latest['high']
        atr = latest['atr']

        signal = None
        reason = ""

        if trend_direction == 'bullish':
            # Pullback buy: low touched EMA20 but closed above it
            if low <= ema20 * 1.001 and price > ema20:
                signal = 'buy'
                reason = f"H4 bullish + H1 pullback to EMA20 (price={price:.0f}, ema20={ema20:.0f})"
            # Breakout buy: closed above previous candle high, above EMA20
            elif price > prev['high'] and price > ema20:
                signal = 'buy'
                reason = f"H4 bullish + H1 breakout (price={price:.0f} > prev high={prev['high']:.0f})"

        elif trend_direction == 'bearish':
            # Pullback sell: high touched EMA20 but closed below it
            if high >= ema20 * 0.999 and price < ema20:
                signal = 'sell'
                reason = f"H4 bearish + H1 pullback to EMA20 (price={price:.0f}, ema20={ema20:.0f})"
            # Breakdown sell: closed below previous candle low, below EMA20
            elif price < prev['low'] and price < ema20:
                signal = 'sell'
                reason = f"H4 bearish + H1 breakdown (price={price:.0f} < prev low={prev['low']:.0f})"

        return signal, reason

    def _calc_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
