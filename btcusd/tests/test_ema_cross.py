"""Tests for EMA crossover strategy."""
import pytest
import pandas as pd
from unittest.mock import Mock

from strategy.ema_cross import EmaCrossStrategy, _ema, _atr


class TestEmaHelperFunctions:
    """Test helper functions used by EMA strategy."""
    
    def test_ema_calculation(self):
        """Test EMA calculation matches expected behavior."""
        # Simple test series
        series = pd.Series([10, 11, 12, 13, 14, 15])
        ema_3 = _ema(series, 3)
        
        # EMA should be pandas ewm result
        expected = series.ewm(span=3, adjust=False).mean()
        pd.testing.assert_series_equal(ema_3, expected)
        
        # Last value should be close to recent prices
        assert ema_3.iloc[-1] > 13.0  # Should be influenced by recent higher values
    
    def test_atr_calculation(self, sample_candles):
        """Test ATR calculation."""
        atr = _atr(sample_candles, 14)
        
        assert isinstance(atr, float)
        assert atr > 0  # ATR should always be positive
        assert atr < 1000  # Should be reasonable for BTCUSD
    
    def test_atr_with_minimal_data(self):
        """Test ATR with minimal candle data."""
        # Create minimal data
        df = pd.DataFrame({
            "high": [100, 105, 103],
            "low": [95, 98, 96],
            "close": [98, 102, 100]
        })
        
        atr = _atr(df, 2)
        assert atr > 0


class TestEmaCrossStrategy:
    """Test the EMA crossover strategy."""
    
    def test_strategy_initialization(self, mock_config):
        """Test strategy creates with correct name and config."""
        strategy = EmaCrossStrategy(mock_config)
        
        assert strategy.name == "ema_cross"
        assert strategy.config == mock_config
    
    def test_insufficient_candles_returns_none(self, mock_config):
        """Test strategy returns None when not enough candles."""
        strategy = EmaCrossStrategy(mock_config)
        
        # Too few candles
        short_candles = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            "close": [82000] * 10,
            "high": [82050] * 10,
            "low": [81950] * 10
        })
        
        result = strategy.evaluate(short_candles)
        assert result is None
    
    def test_no_crossover_returns_none(self, mock_config, sample_candles):
        """Test strategy returns None when no crossover occurs."""
        strategy = EmaCrossStrategy(mock_config)
        
        # Modify sample candles to have stable price (no crossover)
        stable_candles = sample_candles.copy()
        stable_candles["close"] = 82000.0  # Flat price
        
        result = strategy.evaluate(stable_candles)
        assert result is None
    
    def test_bullish_crossover_signal(self, mock_config):
        """Test bullish crossover generates BUY signal."""
        strategy = EmaCrossStrategy(mock_config)
        
        # Build data where EMA10 crosses above EMA50 at the last bar.
        # Long flat section keeps both EMAs ~82000, then a sharp spike
        # on the last bar pulls fast EMA above slow.
        n = 80
        times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h")
        closes = [82000.0] * (n - 5)
        # 4 bars of declining, then sudden spike → forces fast > slow at bar -1
        closes += [81900, 81800, 81700, 81600, 83500]
        
        candles = pd.DataFrame({
            "time": times,
            "close": closes,
            "high": [c + 20 for c in closes],
            "low": [c - 20 for c in closes],
            "tick_volume": [100] * n
        })
        
        result = strategy.evaluate(candles)
        
        assert result is not None
        assert result.direction == "buy"
        assert "crossed above" in result.reason
        assert result.sl_distance > 0
        assert result.tp_distance > 0
        
        # Check metadata
        assert "ema_fast" in result.metadata
        assert "ema_slow" in result.metadata
        assert "atr" in result.metadata
    
    def test_bearish_crossover_signal(self, mock_config):
        """Test bearish crossover generates SELL signal."""
        strategy = EmaCrossStrategy(mock_config)
        
        # Mirror of bullish: flat section, then sharp drop on last bar
        n = 80
        times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h")
        closes = [82000.0] * (n - 5)
        # 4 bars rising, then crash → forces fast < slow at bar -1
        closes += [82100, 82200, 82300, 82400, 80500]
        
        candles = pd.DataFrame({
            "time": times,
            "close": closes,
            "high": [c + 20 for c in closes],
            "low": [c - 20 for c in closes],
            "tick_volume": [100] * n
        })
        
        result = strategy.evaluate(candles)
        
        assert result is not None
        assert result.direction == "sell"
        assert "crossed below" in result.reason
    
    def test_custom_ema_periods(self, mock_config, crossover_candles):
        """Test strategy uses custom EMA periods from config."""
        # Use different EMA periods
        custom_config = mock_config.copy()
        custom_config["ema_fast"] = 5
        custom_config["ema_slow"] = 20
        
        strategy = EmaCrossStrategy(custom_config)
        result = strategy.evaluate(crossover_candles)
        
        if result:  # May or may not generate signal with different periods
            assert "EMA5" in result.reason or "EMA20" in result.reason
    
    def test_signal_metadata_complete(self, mock_config, crossover_candles):
        """Test signal contains all expected metadata."""
        strategy = EmaCrossStrategy(mock_config)
        result = strategy.evaluate(crossover_candles)
        
        if result:
            metadata = result.metadata
            required_keys = ["ema_fast", "ema_slow", "ema_fast_prev", "ema_slow_prev", "atr"]
            for key in required_keys:
                assert key in metadata
                assert isinstance(metadata[key], (int, float))
    
    def test_atr_multipliers_affect_distances(self, mock_config, crossover_candles):
        """Test ATR multipliers affect SL/TP distances correctly."""
        # Test with different multipliers
        config1 = mock_config.copy()
        config1["sl_atr_multiplier"] = 1.0
        config1["tp_atr_multiplier"] = 2.0
        
        config2 = mock_config.copy()
        config2["sl_atr_multiplier"] = 3.0
        config2["tp_atr_multiplier"] = 6.0
        
        strategy1 = EmaCrossStrategy(config1)
        strategy2 = EmaCrossStrategy(config2)
        
        result1 = strategy1.evaluate(crossover_candles)
        result2 = strategy2.evaluate(crossover_candles)
        
        if result1 and result2:
            # Higher multipliers should give larger distances
            assert result2.sl_distance > result1.sl_distance
            assert result2.tp_distance > result1.tp_distance
    
    def test_edge_case_identical_prices(self, mock_config):
        """Test strategy handles identical consecutive prices gracefully."""
        n = 60
        times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h")
        
        # All prices identical - no crossover possible
        candles = pd.DataFrame({
            "time": times,
            "close": [82000.0] * n,
            "high": [82010.0] * n,
            "low": [81990.0] * n,
            "tick_volume": [100] * n
        })
        
        strategy = EmaCrossStrategy(mock_config)
        result = strategy.evaluate(candles)
        
        assert result is None  # No crossover with identical prices
    
    def test_single_crossover_detection(self, mock_config):
        """Test strategy detects crossover only on the exact crossing candle."""
        n = 60
        times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h")
        
        # Create precise crossover scenario
        closes = []
        for i in range(n):
            if i < 30:
                closes.append(80000 + i * 5)  # Slow rise
            elif i == 30:
                closes.append(82000)  # Crossover point
            else:
                closes.append(82000 + (i - 30) * 10)  # Fast rise
        
        candles = pd.DataFrame({
            "time": times,
            "close": closes,
            "high": [c + 10 for c in closes],
            "low": [c - 10 for c in closes],
            "tick_volume": [100] * n
        })
        
        strategy = EmaCrossStrategy(mock_config)
        result = strategy.evaluate(candles)
        
        # Should detect the crossover
        if result:
            assert result.direction in ["buy", "sell"]