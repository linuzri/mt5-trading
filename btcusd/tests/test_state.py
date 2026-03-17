"""Tests for state persistence and management."""
import pytest
import json
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path

from utils.state import State, DEFAULT_STATE


class TestStatePersistence:
    """Test state loading, saving, and persistence."""
    
    def test_state_initialization_new_file(self):
        """Test initializing state when file doesn't exist."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        # Delete the file so it doesn't exist
        os.unlink(temp_path)
        
        try:
            state = State(temp_path)
            
            # Should have default values
            assert state.get("daily_trade_count") == 0
            assert state.get("weekly_trade_count") == 0
            assert state.get("last_trade_ts") is None
            assert state.get("tracked_positions") == {}
            
            # File should be created on first save
            assert not Path(temp_path).exists()
            state.save()
            assert Path(temp_path).exists()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_state_load_existing_file(self, temp_state_file):
        """Test loading state from existing file."""
        # Modify the temp file
        test_data = {
            "daily_trade_count": 5,
            "weekly_trade_count": 15,
            "last_trade_ts": "2024-01-01T10:00:00+00:00",
            "tracked_positions": {"123": {"direction": "buy"}}
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(test_data, f)
        
        state = State(temp_state_file)
        
        assert state.get("daily_trade_count") == 5
        assert state.get("weekly_trade_count") == 15
        assert state.get("last_trade_ts") == "2024-01-01T10:00:00+00:00"
        assert state.get("tracked_positions") == {"123": {"direction": "buy"}}
    
    def test_state_load_corrupted_file(self):
        """Test handling corrupted state file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        try:
            state = State(temp_path)
            
            # Should fall back to defaults
            assert state.get("daily_trade_count") == 0
            assert state.get("weekly_trade_count") == 0
        finally:
            os.unlink(temp_path)
    
    def test_state_save_and_reload(self, temp_state_file):
        """Test saving state and reloading it."""
        state1 = State(temp_state_file)
        
        # Modify state
        state1.set("daily_trade_count", 10)
        state1.set("custom_key", "custom_value")
        state1.save()
        
        # Create new state instance from same file
        state2 = State(temp_state_file)
        
        assert state2.get("daily_trade_count") == 10
        assert state2.get("custom_key") == "custom_value"
    
    def test_state_upgrade_missing_tracked_positions(self, temp_state_file):
        """Test upgrading old state format without tracked_positions."""
        # Create old format without tracked_positions
        old_data = {
            "daily_trade_count": 5,
            "weekly_trade_count": 10
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(old_data, f)
        
        state = State(temp_state_file)
        
        # Should have tracked_positions added
        assert "tracked_positions" in state.as_dict()
        assert state.get("tracked_positions") == {}


class TestStateOperations:
    """Test state getter/setter operations."""
    
    def test_get_set_basic(self, mock_state):
        """Test basic get/set operations."""
        mock_state.set("test_key", "test_value")
        
        assert mock_state.get("test_key") == "test_value"
    
    def test_get_with_default(self, mock_state):
        """Test get with default value."""
        assert mock_state.get("nonexistent", "default") == "default"
        assert mock_state.get("nonexistent") is None
    
    def test_as_dict_returns_copy(self, mock_state):
        """Test as_dict returns dict representation."""
        mock_state.set("key1", "value1")
        mock_state.set("key2", "value2")
        
        data = mock_state.as_dict()
        
        assert isinstance(data, dict)
        assert "key1" in data
        assert "key2" in data
        assert data["key1"] == "value1"


class TestTradeCounters:
    """Test daily/weekly trade counting and resets."""
    
    def test_reset_daily_if_needed_new_date(self, mock_state):
        """Test daily reset when date changes."""
        # Set some initial values
        mock_state.set("daily_trade_count", 10)
        mock_state.set("myt_date", "2024-01-01")
        
        # Reset for new date
        mock_state.reset_daily_if_needed("2024-01-02")
        
        assert mock_state.get("daily_trade_count") == 0
        assert mock_state.get("myt_date") == "2024-01-02"
    
    def test_reset_daily_if_needed_same_date(self, mock_state):
        """Test daily reset doesn't happen for same date."""
        mock_state.set("daily_trade_count", 10)
        mock_state.set("myt_date", "2024-01-01")
        
        # Try to reset with same date
        mock_state.reset_daily_if_needed("2024-01-01")
        
        # Should not reset
        assert mock_state.get("daily_trade_count") == 10
        assert mock_state.get("myt_date") == "2024-01-01"
    
    def test_reset_weekly_if_needed_new_week(self, mock_state):
        """Test weekly reset when week changes."""
        mock_state.set("weekly_trade_count", 25)
        mock_state.set("myt_week_iso", "2024-W01")
        
        # Reset for new week
        mock_state.reset_weekly_if_needed("2024-W02")
        
        assert mock_state.get("weekly_trade_count") == 0
        assert mock_state.get("myt_week_iso") == "2024-W02"
    
    def test_reset_weekly_if_needed_same_week(self, mock_state):
        """Test weekly reset doesn't happen for same week."""
        mock_state.set("weekly_trade_count", 25)
        mock_state.set("myt_week_iso", "2024-W01")
        
        # Try to reset with same week
        mock_state.reset_weekly_if_needed("2024-W01")
        
        # Should not reset
        assert mock_state.get("weekly_trade_count") == 25
        assert mock_state.get("myt_week_iso") == "2024-W01"
    
    def test_record_trade_increments_counters(self, mock_state):
        """Test record_trade increments counters and sets timestamp."""
        initial_daily = mock_state.get("daily_trade_count", 0)
        initial_weekly = mock_state.get("weekly_trade_count", 0)
        
        mock_state.record_trade()
        
        assert mock_state.get("daily_trade_count") == initial_daily + 1
        assert mock_state.get("weekly_trade_count") == initial_weekly + 1
        
        # Should set last trade timestamp
        last_ts = mock_state.get("last_trade_ts")
        assert last_ts is not None
        assert isinstance(last_ts, str)  # ISO format
        
        # Should be recent timestamp
        ts_dt = datetime.fromisoformat(last_ts)
        now = datetime.now(timezone.utc)
        assert (now - ts_dt).total_seconds() < 5  # Within 5 seconds


class TestPositionTracking:
    """Test position tracking functionality."""
    
    def test_track_position_basic(self, mock_state):
        """Test basic position tracking."""
        ticket = 123456
        info = {
            "direction": "buy",
            "entry_price": 82000.0,
            "open_time": datetime.now(timezone.utc).isoformat(),
            "lot": 0.1
        }
        
        mock_state.track_position(ticket, info)
        
        tracked = mock_state.get_tracked_positions()
        assert str(ticket) in tracked
        assert tracked[str(ticket)] == info
    
    def test_track_multiple_positions(self, mock_state):
        """Test tracking multiple positions."""
        positions = {
            123456: {"direction": "buy", "lot": 0.1},
            789012: {"direction": "sell", "lot": 0.2},
            345678: {"direction": "buy", "lot": 0.15}
        }
        
        for ticket, info in positions.items():
            mock_state.track_position(ticket, info)
        
        tracked = mock_state.get_tracked_positions()
        assert len(tracked) == 3
        
        for ticket, info in positions.items():
            assert str(ticket) in tracked
            assert tracked[str(ticket)] == info
    
    def test_untrack_position_exists(self, mock_state):
        """Test untracking existing position."""
        ticket = 123456
        info = {"direction": "buy", "lot": 0.1}
        
        mock_state.track_position(ticket, info)
        
        # Untrack it
        returned_info = mock_state.untrack_position(ticket)
        
        assert returned_info == info
        
        # Should be removed from tracked positions
        tracked = mock_state.get_tracked_positions()
        assert str(ticket) not in tracked
    
    def test_untrack_position_not_exists(self, mock_state):
        """Test untracking non-existent position."""
        returned_info = mock_state.untrack_position(999999)
        
        assert returned_info is None
    
    def test_get_tracked_positions_empty(self, mock_state):
        """Test getting tracked positions when empty."""
        tracked = mock_state.get_tracked_positions()
        
        assert isinstance(tracked, dict)
        assert len(tracked) == 0
    
    def test_get_tracked_positions_returns_copy(self, mock_state):
        """Test that get_tracked_positions returns a copy."""
        ticket = 123456
        info = {"direction": "buy"}
        
        mock_state.track_position(ticket, info)
        
        tracked1 = mock_state.get_tracked_positions()
        tracked2 = mock_state.get_tracked_positions()
        
        # Should be different objects (copies)
        assert tracked1 is not tracked2
        assert tracked1 == tracked2
        
        # Modifying one shouldn't affect the other
        tracked1["999999"] = {"direction": "sell"}
        assert "999999" not in tracked2
    
    def test_position_tracking_persistence(self, temp_state_file):
        """Test position tracking survives save/load cycle."""
        state1 = State(temp_state_file)
        
        # Track a position
        state1.track_position(123456, {
            "direction": "buy",
            "entry_price": 82000.0,
            "lot": 0.1
        })
        state1.save()
        
        # Load fresh state
        state2 = State(temp_state_file)
        tracked = state2.get_tracked_positions()
        
        assert "123456" in tracked
        assert tracked["123456"]["direction"] == "buy"
        assert tracked["123456"]["entry_price"] == 82000.0
    
    def test_position_tracking_with_string_ticket(self, mock_state):
        """Test position tracking handles string tickets correctly."""
        # Sometimes ticket might come as string
        ticket_str = "123456"
        info = {"direction": "buy"}
        
        mock_state.track_position(ticket_str, info)
        
        tracked = mock_state.get_tracked_positions()
        assert "123456" in tracked
        
        # Should be able to untrack with int as well
        returned = mock_state.untrack_position(123456)
        assert returned == info


class TestStateFileHandling:
    """Test file I/O and error handling."""
    
    def test_save_creates_directory(self):
        """Test save creates parent directories."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "subdir", "state.json")
            
            state = State(nested_path)
            state.set("test", "value")
            state.save()
            
            assert os.path.exists(nested_path)
            
            # Should be able to read it back
            state2 = State(nested_path)
            assert state2.get("test") == "value"
    
    def test_save_error_handling(self, mock_state, monkeypatch):
        """Test save handles write errors gracefully."""
        # Mock open to raise an exception
        import builtins
        original_open = builtins.open
        
        def failing_open(*args, **kwargs):
            if "w" in kwargs.get("mode", ""):
                raise PermissionError("Mocked permission error")
            return original_open(*args, **kwargs)
        
        monkeypatch.setattr(builtins, "open", failing_open)
        
        # Should not raise exception
        mock_state.save()  # Should handle the error silently
    
    def test_default_state_values(self):
        """Test DEFAULT_STATE contains expected structure."""
        assert "daily_trade_count" in DEFAULT_STATE
        assert "weekly_trade_count" in DEFAULT_STATE
        assert "last_trade_ts" in DEFAULT_STATE
        assert "tracked_positions" in DEFAULT_STATE
        
        assert DEFAULT_STATE["daily_trade_count"] == 0
        assert DEFAULT_STATE["weekly_trade_count"] == 0
        assert DEFAULT_STATE["last_trade_ts"] is None
        assert DEFAULT_STATE["tracked_positions"] == {}