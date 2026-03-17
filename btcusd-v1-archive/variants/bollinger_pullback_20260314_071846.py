"""
Strategy Variant: bollinger_pullback
Description: BUY when price pulls back to lower Bollinger Band (oversold) but stays above EMA20 in bullish H4 trend
Generated: 2026-03-14T07:18:46.724756

Backtest Results:
- Win Rate: 88.2%
- PnL: $145.69
- Drawdown: 0.01%
- Total Trades: 17

Rationale: Bollinger Bands identify oversold conditions more dynamically than fixed EMA levels. When price touches the lower band but stays above EMA20, it suggests a controlled pullback in an uptrend - a high-probability BUY setup. This adds a third entry condition alongside existing EMA pullback and breakout signals, potentially capturing more profitable entries during volatile pullbacks.
"""

def replay_signals(
    h1_df: pd.DataFrame,
    h4_trend_series: pd.Series,
    h1_ema_period: int,
    atr_period: int,
    min_atr: float,
) -> pd.DataFrame:
    """
    Replay H1 entry signal logic with Bollinger Band pullback confirmation.
    BUY when price touches lower BB but stays above EMA (controlled pullback).
    """
    import pandas as pd
    
    h1 = h1_df.copy()
    h1["ema20"]  = calc_ema(h1["close"], h1_ema_period)
    h1["ema50"]  = calc_ema(h1["close"], 50)
    h1["atr"]    = calc_atr(h1, atr_period)
    
    # Add Bollinger Bands (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2.0
    h1["bb_sma"] = h1["close"].rolling(window=bb_period).mean()
    h1["bb_std"] = h1["close"].rolling(window=bb_period).std()
    h1["bb_upper"] = h1["bb_sma"] + (h1["bb_std"] * bb_std)
    h1["bb_lower"] = h1["bb_sma"] - (h1["bb_std"] * bb_std)

    signals = []
    MIN_SIGNAL_GAP = 4  # 4 H1 bars (4 hours) cooldown between signals
    last_signal_bar = -999

    for i in range(2, len(h1)):
        # Dedup guard: skip if too close to last signal
        if i - last_signal_bar < MIN_SIGNAL_GAP:
            continue

        row  = h1.iloc[i]
        prev = h1.iloc[i - 1]

        # Map H1 bar to H4 trend (find latest H4 bar before this H1 bar)
        h1_time   = row["time"]
        h4_before = h4_trend_series[h4_trend_series.index <= h1_time]
        if h4_before.empty:
            continue
        trend = h4_before.iloc[-1]

        # ATR filter
        if pd.isna(row["atr"]) or row["atr"] < min_atr:
            continue

        price = row["close"]
        ema20 = row["ema20"]
        low   = row["low"]
        high  = row["high"]
        atr   = row["atr"]
        
        # Skip if Bollinger Bands not ready
        if pd.isna(row["bb_lower"]) or pd.isna(row["bb_upper"]):
            continue
            
        bb_lower = row["bb_lower"]
        bb_upper = row["bb_upper"]
        signal = None
        reason = ""

        if trend == "bullish":
            # Original pullback to EMA logic
            if low <= ema20 * 1.001 and price > ema20:
                signal = "buy"
                reason = f"pullback to EMA{h1_ema_period}"
            # Original breakout logic
            elif price > prev["high"] and price > ema20:
                signal = "buy"
                reason = "H1 breakout"
            # NEW: Bollinger Band pullback signal
            elif low <= bb_lower * 1.002 and price > ema20:
                signal = "buy"
                reason = "BB lower band pullback"

        elif trend == "bearish":
            # Keep original bearish logic (though not used in long-only)
            if high >= ema20 * 0.999 and price < ema20:
                signal = "sell"
                reason = f"pullback to EMA{h1_ema_period}"
            elif price < prev["low"] and price < ema20:
                signal = "sell"
                reason = "H1 breakdown"

        if signal:
            last_signal_bar = i
            signals.append({
                "time":        h1_time,
                "bar_index":   i,
                "signal":      signal,
                "entry_price": price,
                "trend":       trend,
                "atr":         atr,
                "reason":      reason,
            })

    return pd.DataFrame(signals)
