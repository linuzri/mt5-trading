"""Tests for trade logging enrichment features."""
import pytest
import pandas as pd
from datetime import datetime, timezone
import numpy as np
from strategy.ema_cross import EmaCrossStrategy


def create_crossover_candles():
    """Create candle data that generates a crossover signal."""
    n = 80
    times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h")
    closes = [82000.0] * (n - 5) + [81900, 81800, 81700, 81600, 83500]
    
    return pd.DataFrame({
        "time": times,
        "close": closes,
        "high": [c + 20 for c in closes],
        "low": [c - 20 for c in closes],
        "tick_volume": [100] * n,
        "open": [c - 10 for c in closes]
    })


class TestEMAGapCalculation:
    """Test EMA gap calculation for enrichment."""

    def test_ema_gap_absolute_calculation(self, mock_config):
        """Test that EMA gap absolute value is calculated correctly."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        # Should have enrichment metadata
        assert signal is not None
        assert "ema_gap" in signal.metadata
        
        # Gap should be absolute difference between fast and slow EMA
        gap = signal.metadata["ema_gap"]
        fast_ema = signal.metadata["ema_fast"]
        slow_ema = signal.metadata["ema_slow"]
        expected_gap = round(abs(fast_ema - slow_ema), 2)
        assert gap == expected_gap

    def test_ema_gap_percentage_calculation(self, mock_config):
        """Test that EMA gap percentage is calculated correctly."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        assert "ema_gap_pct" in signal.metadata
        
        # Gap percentage should be (gap / price) * 100
        gap_pct = signal.metadata["ema_gap_pct"]
        gap = signal.metadata["ema_gap"]
        current_price = candles["close"].iloc[-1]
        expected_pct = round((gap / current_price) * 100, 4)
        assert gap_pct == expected_pct

    def test_ema_trend_direction(self, mock_config):
        """Test that EMA trend (converging/diverging) is calculated correctly."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        assert "ema_trend" in signal.metadata
        
        trend = signal.metadata["ema_trend"]
        assert trend in ["converging", "diverging"]


class TestSignalStrength:
    """Test signal strength scoring for enrichment."""

    def test_signal_strength_range(self, mock_config):
        """Test that signal strength is normalized between 0-1."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        assert "signal_strength" in signal.metadata
        
        strength = signal.metadata["signal_strength"]
        assert 0 <= strength <= 1

    def test_signal_strength_calculation(self, mock_config):
        """Test signal strength calculation logic."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        strength = signal.metadata["signal_strength"]
        gap_pct = signal.metadata["ema_gap_pct"]
        
        # Should be normalized gap percentage with reasonable bounds
        # Stronger signals (wider gaps) should have higher scores
        assert isinstance(strength, float)


class TestCandleContext:
    """Test candle body size relative to ATR."""

    def test_candle_body_atr_ratio(self, mock_config):
        """Test candle body to ATR ratio calculation."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        assert "candle_body_atr_ratio" in signal.metadata
        
        ratio = signal.metadata["candle_body_atr_ratio"]
        assert isinstance(ratio, float)
        assert ratio >= 0  # Should be positive

    def test_candle_body_calculation(self, mock_config):
        """Test that candle body size is calculated correctly."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        # Calculate expected body size
        last_candle = candles.iloc[-1]
        expected_body = abs(last_candle["close"] - last_candle["open"])
        
        # Verify calculation indirectly through ratio
        ratio = signal.metadata["candle_body_atr_ratio"]
        atr = signal.metadata["atr"]
        calculated_body = ratio * atr
        
        assert abs(calculated_body - expected_body) < 1.0  # Allow for rounding differences


class TestH4TrendContext:
    """Test higher timeframe (H4) trend context."""

    def test_h4_trend_field_present(self, mock_config):
        """Test that H4 trend field is present in metadata."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        assert "h4_trend" in signal.metadata

    def test_h4_ema_values_present(self, mock_config):
        """Test that H4 EMA values are present in metadata."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        assert "h4_ema_fast" in signal.metadata
        assert "h4_ema_slow" in signal.metadata

    def test_h4_trend_unavailable_fallback(self, mock_config):
        """Test that unavailable H4 data is handled gracefully."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        signal = strategy.evaluate(candles)
        
        assert signal is not None
        # Should handle missing H4 data gracefully
        h4_trend = signal.metadata.get("h4_trend")
        assert h4_trend is not None

    def test_h4_trend_with_h4_candles(self, mock_config):
        """Test H4 trend calculation when H4 data is provided."""
        strategy = EmaCrossStrategy(mock_config)
        candles = create_crossover_candles()
        h4_candles = create_h4_trend_candles()
        
        # Test with H4 candles provided
        signal = strategy.evaluate(candles, h4_candles=h4_candles)
        
        assert signal is not None
        assert signal.metadata["h4_trend"] != "unavailable"
        assert signal.metadata["h4_ema_fast"] > 0
        assert signal.metadata["h4_ema_slow"] > 0


def create_h4_trend_candles():
    """Create H4 candle data showing an uptrend."""
    n = 50
    times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="4h")
    base_price = 82000.0
    
    # Create uptrend data
    closes = []
    for i in range(n):
        closes.append(base_price + i * 100)  # Consistent uptrend
    
    return pd.DataFrame({
        "time": times,
        "close": closes,
        "high": [c + 25 for c in closes],
        "low": [c - 25 for c in closes],
        "tick_volume": [500] * n,
        "open": [c - 10 for c in closes]
    })


class TestEdgeCases:
    """Test edge cases for enrichment calculations."""

    def test_insufficient_candles_for_gap_comparison(self, mock_config):
        """Test behavior with insufficient candles for trend comparison."""
        # Create minimal candle data (just enough for crossover, not for trend)
        n = 60  # Just enough for EMA calculation but limited history
        times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h")
        
        # Create a simple crossover pattern
        closes = [82000.0] * (n - 5) + [82100, 82200, 82300, 82400, 82500]
        opens = [c - 10 for c in closes]
        highs = [max(o, c) + 5 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 5 for o, c in zip(opens, closes)]
        volumes = [100] * n
        
        candles = pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": volumes
        })
        
        strategy = EmaCrossStrategy(mock_config)
        signal = strategy.evaluate(candles)
        
        if signal is not None:
            # Should handle edge case gracefully
            assert "ema_trend" in signal.metadata
            trend = signal.metadata["ema_trend"]
            # Could be "unavailable" or still calculated
            assert trend in ["converging", "diverging", "unavailable"]

    def test_zero_atr_handling(self, mock_config):
        """Test behavior when ATR is zero (flat market)."""
        # Create candles with identical prices (zero volatility)
        n = 60
        times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h")
        price = 82000.0
        
        # All prices identical except last few for crossover
        closes = [price] * (n - 5) + [price + 1, price + 2, price + 3, price + 4, price + 5]
        opens = closes[:]  # Same as close
        highs = closes[:]
        lows = closes[:]
        volumes = [100] * n
        
        candles = pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": volumes
        })
        
        strategy = EmaCrossStrategy(mock_config)
        signal = strategy.evaluate(candles)
        
        # Should handle zero/near-zero ATR gracefully
        if signal is not None:
            assert "candle_body_atr_ratio" in signal.metadata
            ratio = signal.metadata["candle_body_atr_ratio"]
            # Should handle division by zero case
            assert isinstance(ratio, (int, float))

    def test_none_candles_handling(self, mock_config):
        """Test that None candles are handled gracefully."""
        strategy = EmaCrossStrategy(mock_config)
        signal = strategy.evaluate(None)
        
        # Should return None, not crash
        assert signal is None


@pytest.fixture
def h4_candles():
    """Generate sample H4 candle data for testing."""
    n = 50
    times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="4h")
    base_price = 82000.0
    
    # Create trend data
    closes = []
    for i in range(n):
        # Uptrend in H4
        closes.append(base_price + i * 100)
    
    opens = [c - 50 for c in closes]
    highs = [max(o, c) + 25 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 25 for o, c in zip(opens, closes)]
    volumes = [500] * n
    
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": volumes
    })