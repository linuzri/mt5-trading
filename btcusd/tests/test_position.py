"""Tests for position management."""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from core.position import PositionManager


class TestPositionManager:
    """Test position opening, closing, and tracking."""
    
    def test_position_manager_initialization(self, mock_market, mock_config, mock_state):
        """Test PositionManager initializes correctly."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        assert pm.market == mock_market
        assert pm.config == mock_config
        assert pm.state == mock_state
        assert pm.magic == mock_config["magic_number"]
    
    def test_open_buy_position_success(self, mock_market, mock_config, mock_state, mock_mt5):
        """Test opening a BUY position successfully."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Mock successful order
        mock_mt5.order_send.return_value.retcode = 10009  # TRADE_RETCODE_DONE
        mock_mt5.order_send.return_value.order = 123456
        
        ticket = pm.open_position(
            direction="buy",
            lot=0.1,
            sl=81800.0,
            tp=82300.0,
            comment="test-buy"
        )
        
        assert ticket == 123456
        mock_mt5.order_send.assert_called_once()
        
        # Check order request parameters
        call_args = mock_mt5.order_send.call_args[0][0]
        assert call_args["type"] == 0  # ORDER_TYPE_BUY
        assert call_args["volume"] == 0.1
        assert call_args["sl"] == 81800.0
        assert call_args["tp"] == 82300.0
    
    def test_open_sell_position_success(self, mock_market, mock_config, mock_state, mock_mt5):
        """Test opening a SELL position successfully."""
        pm = PositionManager(mock_market, mock_config, mock_state, )
        
        # Mock successful order
        mock_mt5.order_send.return_value.retcode = 10009
        mock_mt5.order_send.return_value.order = 789012
        
        ticket = pm.open_position(
            direction="sell",
            lot=0.2,
            sl=82200.0,
            tp=81700.0,
            comment="test-sell"
        )
        
        assert ticket == 789012
        
        # Check it's a sell order
        call_args = mock_mt5.order_send.call_args[0][0]
        assert call_args["type"] == 1  # ORDER_TYPE_SELL
        assert call_args["volume"] == 0.2
    
    def test_open_position_order_fails(self, mock_market, mock_config, mock_state, mock_mt5):
        """Test handling failed order."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Mock failed order
        mock_mt5.order_send.return_value.retcode = 10013  # Invalid request
        mock_mt5.order_send.return_value.order = None
        
        ticket = pm.open_position("buy", 0.1, 81800.0, 82300.0)
        
        assert ticket is None
    
    def test_open_position_no_tick_data(self, mock_market, mock_config, mock_state, monkeypatch):
        """Test handling missing tick data."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Mock no tick data
        monkeypatch.setattr(mock_market, 'get_tick', lambda: None)
        
        ticket = pm.open_position("buy", 0.1, 81800.0, 82300.0)
        
        assert ticket is None
    
    def test_open_position_dry_run(self, mock_market, mock_config, mock_state, mock_mt5):
        """Test dry run mode doesn't place actual orders."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        ticket = pm.open_position(
            direction="buy",
            lot=0.1,
            sl=81800.0,
            tp=82300.0,
            dry_run=True
        )
        
        assert ticket == -1  # Dry run returns -1
        mock_mt5.order_send.assert_not_called()
    
    def test_close_position_success(self, mock_market, mock_config, mock_state, mock_mt5, mock_position):
        """Test closing a position successfully."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Mock position exists and close succeeds
        mock_mt5.positions_get.return_value = [mock_position]
        mock_mt5.order_send.return_value.retcode = 10009
        
        profit = pm.close_position(123456)
        
        assert profit == mock_position.profit
        mock_mt5.order_send.assert_called_once()
        
        # Check close request
        call_args = mock_mt5.order_send.call_args[0][0]
        assert call_args["position"] == 123456
        assert call_args["type"] == 1  # ORDER_TYPE_SELL (closing BUY)
    
    def test_close_position_not_found(self, mock_market, mock_config, mock_state, mock_mt5):
        """Test closing non-existent position."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # No position found
        mock_mt5.positions_get.return_value = []
        
        profit = pm.close_position(999999)
        
        assert profit is None
        mock_mt5.order_send.assert_not_called()
    
    def test_close_position_dry_run(self, mock_market, mock_config, mock_state, mock_mt5, mock_position):
        """Test dry run close doesn't place actual close order."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        mock_mt5.positions_get.return_value = [mock_position]
        
        profit = pm.close_position(123456, dry_run=True)
        
        assert profit == 0.0  # Dry run returns 0 profit
        mock_mt5.order_send.assert_not_called()
    
    def test_check_closed_positions_detects_closure(self, mock_market, mock_config, mock_state, monkeypatch):
        """Test detection of closed positions."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Add a tracked position
        mock_state.track_position(123456, {
            "direction": "buy",
            "entry_price": 82000.0,
            "open_time": datetime.now(timezone.utc).isoformat(),
            "lot": 0.1,
            "sl": 81800.0,
            "tp": 82300.0
        })
        
        # Mock: position no longer in open positions
        monkeypatch.setattr(mock_market, 'get_positions', lambda: [])
        monkeypatch.setattr(mock_market, 'get_account', lambda: (50000.0, 50000.0))
        
        closed = pm.check_closed_positions(mock_market)
        
        assert len(closed) == 1
        assert closed[0]["ticket"] == 123456
        assert closed[0]["direction"] == "buy"
        assert closed[0]["entry_price"] == 82000.0
    
    def test_check_closed_positions_no_state(self, mock_market, mock_config):
        """Test check_closed_positions with no state manager."""
        pm = PositionManager(mock_market, mock_config, None)
        
        closed = pm.check_closed_positions(mock_market)
        
        assert closed == []
    
    def test_recover_positions_from_mt5(self, mock_market, mock_config, mock_state, mock_position, monkeypatch):
        """Test recovering open positions from MT5 on startup."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Mock MT5 has open position not in state
        monkeypatch.setattr(mock_market, 'get_positions', lambda: [mock_position])
        
        recovered_count = pm.recover_positions(mock_market)
        
        assert recovered_count == 1
        
        # Check position was tracked
        tracked = mock_state.get_tracked_positions()
        assert "123456" in tracked
    
    def test_recover_positions_already_tracked(self, mock_market, mock_config, mock_state, mock_position, monkeypatch):
        """Test recovery ignores already-tracked positions."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Already track the position
        mock_state.track_position(123456, {"direction": "buy"})
        monkeypatch.setattr(mock_market, 'get_positions', lambda: [mock_position])
        
        recovered_count = pm.recover_positions(mock_market)
        
        assert recovered_count == 0  # Nothing new to recover
    
    def test_calculate_sl_tp_buy(self, mock_market, mock_config, mock_state):
        """Test SL/TP calculation for BUY position."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        price = 82000.0
        atr = 100.0
        
        sl, tp = pm.calculate_sl_tp("buy", price, atr)
        
        # BUY: SL below entry, TP above entry
        expected_sl = price - (atr * 2.0)  # sl_atr_multiplier = 2.0
        expected_tp = price + (atr * 3.0)  # tp_atr_multiplier = 3.0
        
        assert sl == expected_sl
        assert tp == expected_tp
    
    def test_calculate_sl_tp_sell(self, mock_market, mock_config, mock_state):
        """Test SL/TP calculation for SELL position."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        price = 82000.0
        atr = 100.0
        
        sl, tp = pm.calculate_sl_tp("sell", price, atr)
        
        # SELL: SL above entry, TP below entry
        expected_sl = price + (atr * 2.0)
        expected_tp = price - (atr * 3.0)
        
        assert sl == expected_sl
        assert tp == expected_tp
    
    def test_position_tracking_in_state(self, mock_market, mock_config, mock_state, mock_mt5):
        """Test that opened positions are tracked in state."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        mock_mt5.order_send.return_value.retcode = 10009
        mock_mt5.order_send.return_value.order = 555666
        
        ticket = pm.open_position("buy", 0.1, 81800.0, 82300.0)
        
        # Check position was tracked
        tracked = mock_state.get_tracked_positions()
        assert "555666" in tracked
        assert tracked["555666"]["direction"] == "buy"
    
    def test_build_close_data_with_deal_history(self, mock_market, mock_config, mock_state, mock_mt5):
        """Test enriched close data includes deal history."""
        pm = PositionManager(mock_market, mock_config, mock_state)
        
        # Mock deal history
        deal_mock = Mock()
        deal_mock.entry = 1  # DEAL_ENTRY_OUT (closing)
        deal_mock.price = 82100.0
        deal_mock.profit = 50.0
        deal_mock.reason = 4  # DEAL_REASON_TP
        mock_mt5.history_deals_get.return_value = [deal_mock]
        
        info = {
            "direction": "buy",
            "entry_price": 82000.0,
            "open_time": datetime.now(timezone.utc).isoformat(),
            "lot": 0.1,
            "sl": 81800.0,
            "tp": 82300.0
        }
        
        close_data = pm._build_close_data(123456, info, mock_market)
        
        assert close_data["close_price"] == 82100.0
        assert close_data["profit"] == 50.0
        assert close_data["close_reason"] == "take_profit"
        assert close_data["profit_pips"] > 0  # BUY position closed higher