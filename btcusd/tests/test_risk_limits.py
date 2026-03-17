"""Tests for risk limits and controls."""
import pytest
from datetime import datetime, timezone, timedelta

from risk.limits import Limits


class TestRiskLimits:
    """Test risk limit checks and controls."""
    
    def test_limits_initialization(self, mock_config):
        """Test Limits initializes with config."""
        limits = Limits(mock_config)
        
        assert limits.config == mock_config
    
    def test_check_all_limits_pass(self, mock_config):
        """Test when all limits pass."""
        limits = Limits(mock_config)
        
        # Clean state - no trades, no positions
        state = {
            "daily_trade_count": 0,
            "weekly_trade_count": 0,
            "last_trade_ts": None
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is True
        assert reason == "ok"
    
    def test_max_open_positions_limit(self, mock_config):
        """Test max open positions limit."""
        limits = Limits(mock_config)
        
        state = {
            "daily_trade_count": 0,
            "weekly_trade_count": 0,
            "last_trade_ts": None
        }
        
        # At the limit (3 open, max is 3)
        allowed, reason = limits.check(state, open_position_count=3)
        
        assert allowed is False
        assert "max_open_positions" in reason
        assert "(3/3)" in reason
    
    def test_max_open_positions_under_limit(self, mock_config):
        """Test under max open positions limit."""
        limits = Limits(mock_config)
        
        state = {
            "daily_trade_count": 0,
            "weekly_trade_count": 0,
            "last_trade_ts": None
        }
        
        # Under limit (2 open, max is 3)
        allowed, reason = limits.check(state, open_position_count=2)
        
        assert allowed is True
        assert reason == "ok"
    
    def test_daily_trade_limit(self, mock_config):
        """Test daily trade count limit."""
        limits = Limits(mock_config)
        
        # At daily limit
        state = {
            "daily_trade_count": 50,  # max_daily_trades = 50
            "weekly_trade_count": 50,
            "last_trade_ts": None
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is False
        assert "max_daily_trades" in reason
        assert "(50/50)" in reason
    
    def test_daily_trade_under_limit(self, mock_config):
        """Test under daily trade limit."""
        limits = Limits(mock_config)
        
        state = {
            "daily_trade_count": 25,  # Under limit of 50
            "weekly_trade_count": 25,
            "last_trade_ts": None
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is True
        assert reason == "ok"
    
    def test_weekly_trade_limit(self, mock_config):
        """Test weekly trade count limit."""
        limits = Limits(mock_config)
        
        # At weekly limit
        state = {
            "daily_trade_count": 10,
            "weekly_trade_count": 200,  # max_weekly_trades = 200
            "last_trade_ts": None
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is False
        assert "max_weekly_trades" in reason
        assert "(200/200)" in reason
    
    def test_cooldown_period_active(self, mock_config):
        """Test cooldown period blocking new trades."""
        limits = Limits(mock_config)
        
        # Last trade was 2 minutes ago (cooldown is 5 minutes)
        last_trade = datetime.now(timezone.utc) - timedelta(minutes=2)
        
        state = {
            "daily_trade_count": 5,
            "weekly_trade_count": 10,
            "last_trade_ts": last_trade.isoformat()
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is False
        assert "cooldown_active" in reason
        assert "remaining=" in reason
    
    def test_cooldown_period_expired(self, mock_config):
        """Test cooldown period has expired."""
        limits = Limits(mock_config)
        
        # Last trade was 10 minutes ago (cooldown is 5 minutes)
        last_trade = datetime.now(timezone.utc) - timedelta(minutes=10)
        
        state = {
            "daily_trade_count": 5,
            "weekly_trade_count": 10,
            "last_trade_ts": last_trade.isoformat()
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is True
        assert reason == "ok"
    
    def test_no_last_trade_timestamp(self, mock_config):
        """Test when there's no last trade timestamp (first trade)."""
        limits = Limits(mock_config)
        
        state = {
            "daily_trade_count": 0,
            "weekly_trade_count": 0,
            "last_trade_ts": None  # First trade ever
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is True
        assert reason == "ok"
    
    def test_custom_limits_from_config(self):
        """Test custom limit values from config."""
        custom_config = {
            "max_open_positions": 1,
            "max_daily_trades": 5,
            "max_weekly_trades": 20,
            "cooldown_seconds": 60  # 1 minute
        }
        
        limits = Limits(custom_config)
        
        # Test custom max open positions
        state = {"daily_trade_count": 0, "weekly_trade_count": 0, "last_trade_ts": None}
        allowed, reason = limits.check(state, open_position_count=1)
        assert allowed is False
        assert "(1/1)" in reason
        
        # Test custom daily limit
        state["daily_trade_count"] = 5
        allowed, reason = limits.check(state, open_position_count=0)
        assert allowed is False
        assert "(5/5)" in reason
        
        # Test custom weekly limit
        state["daily_trade_count"] = 1
        state["weekly_trade_count"] = 20
        allowed, reason = limits.check(state, open_position_count=0)
        assert allowed is False
        assert "(20/20)" in reason
    
    def test_multiple_violations_first_one_reported(self, mock_config):
        """Test that first violation is reported when multiple limits violated."""
        limits = Limits(mock_config)
        
        # Violate both open positions and daily trades
        state = {
            "daily_trade_count": 50,  # At limit
            "weekly_trade_count": 50,
            "last_trade_ts": None
        }
        
        # Also at open position limit
        allowed, reason = limits.check(state, open_position_count=3)
        
        assert allowed is False
        # Should report the first check that fails (open positions)
        assert "max_open_positions" in reason
    
    def test_missing_state_keys_default_values(self, mock_config):
        """Test handling of missing keys in state dict."""
        limits = Limits(mock_config)
        
        # Empty state dict
        state = {}
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        # Should not crash and should pass (defaults to 0)
        assert allowed is True
        assert reason == "ok"
    
    def test_cooldown_calculation_precision(self, mock_config):
        """Test cooldown remaining time calculation."""
        limits = Limits(mock_config)
        
        # Exactly 2 minutes ago (cooldown is 5 minutes = 300 seconds)
        last_trade = datetime.now(timezone.utc) - timedelta(seconds=120)
        
        state = {
            "daily_trade_count": 1,
            "weekly_trade_count": 1,
            "last_trade_ts": last_trade.isoformat()
        }
        
        allowed, reason = limits.check(state, open_position_count=0)
        
        assert allowed is False
        # Should have about 180 seconds remaining (300 - 120)
        assert "remaining=180s" in reason or "remaining=179s" in reason
    
    def test_zero_limits_always_block(self):
        """Test that zero limits always block trades."""
        zero_config = {
            "max_open_positions": 0,
            "max_daily_trades": 0,
            "max_weekly_trades": 0,
            "cooldown_seconds": 0
        }
        
        limits = Limits(zero_config)
        state = {"daily_trade_count": 0, "weekly_trade_count": 0, "last_trade_ts": None}
        
        # Should block due to max_open_positions = 0
        allowed, reason = limits.check(state, open_position_count=0)
        assert allowed is False
        assert "max_open_positions" in reason
    
    def test_negative_open_position_count(self, mock_config):
        """Test handling negative open position count."""
        limits = Limits(mock_config)
        state = {"daily_trade_count": 0, "weekly_trade_count": 0, "last_trade_ts": None}
        
        # Negative count should be treated as 0
        allowed, reason = limits.check(state, open_position_count=-1)
        
        assert allowed is True
        assert reason == "ok"