"""Tests for multi-timeframe logging functionality."""
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from core.market import Market
from strategy.ema_cross import EmaCrossStrategy


class TestMultiTimeframeFetching:
    """Test the get_candles_tf method in Market class."""
    
    def test_get_candles_tf_returns_h4_data(self, mock_market, mock_mt5):
        """Test that get_candles_tf can fetch H4 timeframe data."""
        # Setup mock data for H4 candles
        h4_rates = [
            {"time": int(datetime.now(timezone.utc).timestamp()) - 14400 * i, 
             "open": 82000.0 + i*10, "high": 82100.0 + i*10, 
             "low": 81900.0 + i*10, "close": 82050.0 + i*10, 
             "tick_volume": 100}
            for i in range(50)
        ]
        mock_mt5.copy_rates_from_pos.return_value = h4_rates
        
        # Test H4 fetch
        result = mock_market.get_candles_tf("H4", count=50)
        
        assert result is not None
        assert len(result) == 50
        assert list(result.columns) == ["time", "open", "high", "low", "close", "tick_volume"]
        # Verify MT5 was called with H4 timeframe
        mock_mt5.copy_rates_from_pos.assert_called_with(
            mock_market.symbol, mock_mt5.TIMEFRAME_H4, 0, 50
        )
    
    def test_get_candles_tf_returns_d1_data(self, mock_market, mock_mt5):
        """Test that get_candles_tf can fetch D1 timeframe data."""
        # Setup mock data for D1 candles
        d1_rates = [
            {"time": int(datetime.now(timezone.utc).timestamp()) - 86400 * i,
             "open": 82000.0 + i*50, "high": 82200.0 + i*50,
             "low": 81800.0 + i*50, "close": 82100.0 + i*50,
             "tick_volume": 500}
            for i in range(30)
        ]
        mock_mt5.copy_rates_from_pos.return_value = d1_rates
        
        # Test D1 fetch
        result = mock_market.get_candles_tf("D1", count=30)
        
        assert result is not None
        assert len(result) == 30
        assert list(result.columns) == ["time", "open", "high", "low", "close", "tick_volume"]
        # Verify MT5 was called with D1 timeframe
        mock_mt5.copy_rates_from_pos.assert_called_with(
            mock_market.symbol, mock_mt5.TIMEFRAME_D1, 0, 30
        )
    
    def test_get_candles_tf_invalid_timeframe(self, mock_market):
        """Test that invalid timeframe raises appropriate error."""
        with pytest.raises(KeyError):
            mock_market.get_candles_tf("INVALID")
    
    def test_get_candles_tf_mt5_returns_none(self, mock_market, mock_mt5):
        """Test graceful handling when MT5 returns no data."""
        mock_mt5.copy_rates_from_pos.return_value = None
        
        result = mock_market.get_candles_tf("H4")
        
        assert result is None
    
    def test_get_candles_tf_connection_failure(self, mock_market, mock_mt5):
        """Test handling when MT5 connection fails."""
        mock_mt5.terminal_info.return_value = None
        mock_market._connected = False
        
        # Mock connect to fail
        with patch.object(mock_market, 'connect', return_value=False):
            result = mock_market.get_candles_tf("H4")
        
        assert result is None


class TestStrategyD1Analysis:
    """Test D1 trend analysis in EMA cross strategy."""
    
    def test_evaluate_with_d1_candles_computes_trend(self, sample_candles):
        """Test that strategy computes D1 trend when d1_candles provided."""
        strategy = EmaCrossStrategy({"ema_fast": 10, "ema_slow": 50})
        
        # Create D1 candles with clear uptrend
        d1_candles = sample_candles.copy()
        # Modify to create clear uptrend where fast > slow
        d1_candles["close"] = range(80000, 80000 + len(d1_candles) * 100, 100)
        
        # Call evaluate with both H1 and D1 candles
        signal = strategy.evaluate(sample_candles, d1_candles=d1_candles)
        
        # Should have D1 trend metadata
        if signal:
            assert "d1_trend" in signal.metadata
            assert "d1_ema_fast" in signal.metadata  
            assert "d1_ema_slow" in signal.metadata
            assert signal.metadata["d1_trend"] in ["above", "below", "neutral"]
            assert isinstance(signal.metadata["d1_ema_fast"], (int, float))
            assert isinstance(signal.metadata["d1_ema_slow"], (int, float))
    
    def test_evaluate_without_d1_candles_shows_unavailable(self, sample_candles):
        """Test that D1 trend shows 'unavailable' when no D1 data provided."""
        strategy = EmaCrossStrategy({"ema_fast": 10, "ema_slow": 50})
        
        signal = strategy.evaluate(sample_candles)
        
        if signal:
            assert signal.metadata["d1_trend"] == "unavailable"
            assert signal.metadata["d1_ema_fast"] == 0.0
            assert signal.metadata["d1_ema_slow"] == 0.0
    
    def test_evaluate_insufficient_d1_candles_shows_unavailable(self, sample_candles):
        """Test handling when D1 has insufficient candles for EMA calculation."""
        strategy = EmaCrossStrategy({"ema_fast": 10, "ema_slow": 50})
        
        # Create D1 with too few candles
        insufficient_d1 = sample_candles.head(10)  # Only 10 candles, need 50+ for slow EMA
        
        signal = strategy.evaluate(sample_candles, d1_candles=insufficient_d1)
        
        if signal:
            assert signal.metadata["d1_trend"] == "unavailable"
            assert signal.metadata["d1_ema_fast"] == 0.0
            assert signal.metadata["d1_ema_slow"] == 0.0
    
    def test_d1_trend_calculation_above(self, sample_candles):
        """Test D1 trend calculation when fast EMA > slow EMA."""
        strategy = EmaCrossStrategy({"ema_fast": 10, "ema_slow": 50})
        
        # Create D1 candles with fast > slow (uptrend)
        d1_candles = sample_candles.copy()
        d1_candles["close"] = range(80000, 80000 + len(d1_candles) * 200, 200)
        
        signal = strategy.evaluate(sample_candles, d1_candles=d1_candles)
        
        if signal:
            assert signal.metadata["d1_trend"] == "above"
            assert signal.metadata["d1_ema_fast"] > signal.metadata["d1_ema_slow"]
    
    def test_d1_trend_calculation_below(self, sample_candles):
        """Test D1 trend calculation when fast EMA < slow EMA.""" 
        strategy = EmaCrossStrategy({"ema_fast": 10, "ema_slow": 50})
        
        # Create D1 candles with fast < slow (downtrend)
        d1_candles = sample_candles.copy()
        d1_candles["close"] = range(85000, 85000 - len(d1_candles) * 200, -200)
        
        signal = strategy.evaluate(sample_candles, d1_candles=d1_candles)
        
        if signal:
            assert signal.metadata["d1_trend"] == "below"
            assert signal.metadata["d1_ema_fast"] < signal.metadata["d1_ema_slow"]
    
    def test_d1_error_handling(self, sample_candles):
        """Test graceful handling of errors in D1 analysis."""
        strategy = EmaCrossStrategy({"ema_fast": 10, "ema_slow": 50})
        
        # Create malformed D1 data
        bad_d1 = pd.DataFrame({"bad_column": [1, 2, 3]})
        
        signal = strategy.evaluate(sample_candles, d1_candles=bad_d1)
        
        if signal:
            assert signal.metadata["d1_trend"] == "error"


class TestEngineIntegration:
    """Integration tests to verify engine fetches and passes multi-TF candles."""
    
    def test_engine_tick_with_multi_tf_support(self, mock_config, mock_mt5, temp_state_file):
        """Test that modified engine can handle multi-TF fetching without crashing."""
        from core.engine import Engine
        from core.market import Market
        from strategy.ema_cross import EmaCrossStrategy
        from utils.state import State
        from risk.limits import Limits
        from risk.sizing import Sizer
        from notifications.telegram import Telegram
        
        # Setup mock data for different timeframes
        h1_rates = [{"time": int(datetime.now(timezone.utc).timestamp()) - 3600 * i, 
                     "open": 82000.0, "high": 82100.0, "low": 81900.0, "close": 82050.0, "tick_volume": 100}
                    for i in range(100)]
        h4_rates = [{"time": int(datetime.now(timezone.utc).timestamp()) - 14400 * i,
                     "open": 82000.0, "high": 82100.0, "low": 81900.0, "close": 82050.0, "tick_volume": 100}
                    for i in range(50)]
        d1_rates = [{"time": int(datetime.now(timezone.utc).timestamp()) - 86400 * i,
                     "open": 82000.0, "high": 82100.0, "low": 81900.0, "close": 82050.0, "tick_volume": 100}
                    for i in range(30)]
        
        def mock_copy_rates_side_effect(symbol, timeframe, start, count):
            if timeframe == mock_mt5.TIMEFRAME_H1:
                return h1_rates
            elif timeframe == mock_mt5.TIMEFRAME_H4:
                return h4_rates  
            elif timeframe == mock_mt5.TIMEFRAME_D1:
                return d1_rates
            return None
            
        mock_mt5.copy_rates_from_pos.side_effect = mock_copy_rates_side_effect
        
        # Create minimal engine with all required components
        market = Market(mock_config)
        strategy = EmaCrossStrategy(mock_config)
        state = State(temp_state_file)
        limits = Limits(mock_config)
        sizer = Sizer(mock_config)
        telegram = Mock()  # Simple mock
        telegram.send = Mock()
        
        engine = Engine(
            config=mock_config,
            market=market,
            strategy=strategy,
            sizer=sizer,
            limits=limits,
            state=state,
            telegram=telegram,
            dry_run=True
        )
        
        # Mock strategy.evaluate to verify the call signature
        original_evaluate = strategy.evaluate
        evaluate_calls = []
        
        def mock_evaluate(*args, **kwargs):
            evaluate_calls.append((args, kwargs))
            return None  # No signal to avoid position logic
            
        strategy.evaluate = Mock(side_effect=mock_evaluate)
        
        # Run the tick - should not crash and should call evaluate with multi-TF data
        engine._tick()
        
        # Verify that strategy.evaluate was called
        assert len(evaluate_calls) == 1
        args, kwargs = evaluate_calls[0]
        
        # Verify it was called with candles and multi-TF parameters
        assert len(args) == 1  # H1 candles as positional arg
        assert "h4_candles" in kwargs
        assert "d1_candles" in kwargs
        
        # In dry_run mode, multi-TF candles should be None (since we fetch them only when not dry_run)
        # But the call signature should work
        assert kwargs["h4_candles"] is None
        assert kwargs["d1_candles"] is None