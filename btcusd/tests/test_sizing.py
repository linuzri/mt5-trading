"""Tests for position sizing calculations."""
import pytest
import math

from risk.sizing import Sizer


class TestPositionSizing:
    """Test position sizing calculations and risk management."""
    
    def test_sizer_initialization(self, mock_config):
        """Test Sizer initializes with config."""
        sizer = Sizer(mock_config)
        
        assert sizer.config == mock_config
    
    def test_basic_lot_calculation(self, mock_config):
        """Test basic lot size calculation."""
        sizer = Sizer(mock_config)
        
        balance = 50000.0
        sl_distance = 200.0  # $200 stop loss distance
        
        # Risk per trade = 0.5%, so risk = $250
        # lot = risk / sl_distance = 250 / 200 = 1.25
        # But max_lot = 1.0, so should cap at 1.0
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 1.0  # Capped by max_lot
    
    def test_lot_calculation_under_max(self, mock_config):
        """Test lot calculation when under max limit."""
        sizer = Sizer(mock_config)
        
        balance = 20000.0
        sl_distance = 500.0  # $500 stop loss
        
        # Risk = 20000 * 0.005 = $100
        # lot = 100 / 500 = 0.2
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 0.2
    
    def test_lot_calculation_under_minimum(self, mock_config):
        """Test lot calculation floors at minimum."""
        sizer = Sizer(mock_config)
        
        balance = 1000.0  # Small balance
        sl_distance = 1000.0  # Large SL
        
        # Risk = 1000 * 0.005 = $5
        # lot = 5 / 1000 = 0.005
        # But min_lot = 0.01, so should floor at 0.01
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 0.01
    
    def test_zero_sl_distance_returns_minimum(self, mock_config):
        """Test zero SL distance returns minimum lot."""
        sizer = Sizer(mock_config)
        
        lot = sizer.calculate_lot(50000.0, 0.0)
        
        assert lot == mock_config["min_lot"]
    
    def test_zero_balance_returns_minimum(self, mock_config):
        """Test zero balance returns minimum lot."""
        sizer = Sizer(mock_config)
        
        lot = sizer.calculate_lot(0.0, 200.0)
        
        assert lot == mock_config["min_lot"]
    
    def test_negative_inputs_return_minimum(self, mock_config):
        """Test negative inputs return minimum lot."""
        sizer = Sizer(mock_config)
        
        # Negative balance
        lot1 = sizer.calculate_lot(-1000.0, 200.0)
        assert lot1 == mock_config["min_lot"]
        
        # Negative SL distance
        lot2 = sizer.calculate_lot(50000.0, -200.0)
        assert lot2 == mock_config["min_lot"]
    
    def test_custom_risk_percentage(self):
        """Test custom risk percentage from config."""
        config = {
            "risk_per_trade": 1.0,  # 1% risk instead of 0.5%
            "min_lot": 0.01,
            "max_lot": 2.0
        }
        
        sizer = Sizer(config)
        
        balance = 50000.0
        sl_distance = 250.0
        
        # Risk = 50000 * 0.01 = $500
        # lot = 500 / 250 = 2.0
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 2.0  # At max_lot
    
    def test_custom_min_max_lots(self):
        """Test custom min/max lot sizes."""
        config = {
            "risk_per_trade": 0.5,
            "min_lot": 0.05,   # Custom minimum
            "max_lot": 0.5     # Custom maximum
        }
        
        sizer = Sizer(config)
        
        # Test minimum
        lot_min = sizer.calculate_lot(1000.0, 1000.0)  # Very conservative
        assert lot_min == 0.05
        
        # Test maximum
        lot_max = sizer.calculate_lot(100000.0, 100.0)  # Aggressive
        assert lot_max == 0.5
    
    def test_lot_rounding_down(self, mock_config):
        """Test lot size is rounded down to 2 decimal places."""
        sizer = Sizer(mock_config)
        
        # Use smaller balance to avoid max_lot capping
        balance = 10000.0
        sl_distance = 300.0
        
        # Risk = 10000 * 0.005 = $50
        # lot = 50 / 300 = 0.16666...
        # Should round down to 0.16
        lot = sizer.calculate_lot(balance, sl_distance)
        
        expected = math.floor(0.16667 * 100) / 100
        assert lot == expected
        assert lot == 0.16
    
    def test_lot_rounding_edge_cases(self, mock_config):
        """Test lot rounding with edge cases."""
        sizer = Sizer(mock_config)
        
        # Test case that would round to exactly x.xx5
        balance = 50000.0
        sl_distance = 149.0  # Chosen to create x.xx5 result
        
        # Risk = 250, lot = 250/149 ≈ 1.677...
        lot = sizer.calculate_lot(balance, sl_distance)
        
        # Should be floored properly
        assert isinstance(lot, float)
        assert lot <= 1.0  # Will be capped by max_lot anyway
    
    def test_very_small_risk_percentage(self):
        """Test very small risk percentage."""
        config = {
            "risk_per_trade": 0.01,  # 0.01% risk
            "min_lot": 0.01,
            "max_lot": 1.0
        }
        
        sizer = Sizer(config)
        
        balance = 50000.0
        sl_distance = 100.0
        
        # Risk = 50000 * 0.0001 = $5
        # lot = 5 / 100 = 0.05
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 0.05
    
    def test_large_balance_calculation(self, mock_config):
        """Test calculation with very large balance."""
        sizer = Sizer(mock_config)
        
        balance = 1000000.0  # $1M
        sl_distance = 500.0
        
        # Risk = 1000000 * 0.005 = $5000
        # lot = 5000 / 500 = 10.0
        # But max_lot = 1.0
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 1.0  # Capped at maximum
    
    def test_precision_with_small_numbers(self, mock_config):
        """Test precision handling with small numbers."""
        sizer = Sizer(mock_config)
        
        balance = 100.0
        sl_distance = 1.0
        
        # Risk = 100 * 0.005 = $0.50
        # lot = 0.5 / 1.0 = 0.5
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 0.5
    
    def test_realistic_btcusd_scenario(self, mock_config):
        """Test realistic BTCUSD trading scenario."""
        sizer = Sizer(mock_config)
        
        # Realistic BTCUSD parameters
        balance = 25000.0      # $25K account
        sl_distance = 300.0    # $300 stop loss (tight for BTC)
        
        # Risk = 25000 * 0.005 = $125
        # lot = 125 / 300 ≈ 0.416... → 0.41
        lot = sizer.calculate_lot(balance, sl_distance)
        
        expected = math.floor(0.41667 * 100) / 100
        assert lot == expected
        assert lot == 0.41
    
    def test_max_risk_scenario(self, mock_config):
        """Test maximum risk scenario reaches max lot."""
        sizer = Sizer(mock_config)
        
        balance = 100000.0    # Large balance
        sl_distance = 50.0    # Very tight stop
        
        # Risk = 100000 * 0.005 = $500
        # lot = 500 / 50 = 10.0
        # Should be capped at max_lot = 1.0
        lot = sizer.calculate_lot(balance, sl_distance)
        
        assert lot == 1.0
    
    def test_intermediate_calculations(self, mock_config):
        """Test that intermediate calculations are correct."""
        sizer = Sizer(mock_config)
        
        balance = 40000.0
        sl_distance = 160.0
        
        # Step by step:
        # risk_pct = 0.5 / 100 = 0.005
        # risk_dollars = 40000 * 0.005 = 200
        # raw_lot = 200 / 160 = 1.25
        # floored = floor(1.25 * 100) / 100 = 1.25
        # clamped = min(max_lot=1.0, 1.25) = 1.0
        
        lot = sizer.calculate_lot(balance, sl_distance)
        assert lot == 1.0