"""Pytest configuration and shared fixtures for BTCUSD trading bot tests."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pandas as pd
import pytest


@pytest.fixture
def mock_mt5(monkeypatch):
    """Mock the entire MetaTrader5 module to avoid needing real MT5 connection."""
    mock = Mock()
    
    # Terminal and connection mocks
    mock.terminal_info.return_value = Mock()  # Non-None means connected
    mock.initialize.return_value = True
    mock.symbol_select.return_value = True
    mock.shutdown.return_value = None
    mock.last_error.return_value = (0, "Success")
    
    # Constants
    mock.TIMEFRAME_H1 = 16385
    mock.TIMEFRAME_H4 = 16388
    mock.TIMEFRAME_D1 = 16408
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.TRADE_ACTION_DEAL = 1
    mock.ORDER_TIME_GTC = 0
    mock.ORDER_FILLING_IOC = 1
    mock.TRADE_RETCODE_DONE = 10009
    mock.SYMBOL_TRADE_MODE_FULL = 0
    
    # Account info mock
    account_mock = Mock()
    account_mock.login = 12345
    account_mock.balance = 50000.0
    account_mock.equity = 50000.0
    mock.account_info.return_value = account_mock
    
    # Symbol info mock
    symbol_mock = Mock()
    symbol_mock.visible = True
    symbol_mock.trade_mode = 0
    symbol_mock.filling_mode = 2  # IOC
    mock.symbol_info.return_value = symbol_mock
    
    # Tick mock
    tick_mock = Mock()
    tick_mock.bid = 82000.0
    tick_mock.ask = 82005.0
    mock.symbol_info_tick.return_value = tick_mock
    
    # Order send mock - returns successful result
    result_mock = Mock()
    result_mock.retcode = 10009  # TRADE_RETCODE_DONE
    result_mock.order = 123456  # Ticket number
    result_mock.comment = "Success"
    mock.order_send.return_value = result_mock
    
    # Positions mock - empty by default
    mock.positions_get.return_value = []
    
    # Rates/candles mock
    mock.copy_rates_from_pos.return_value = None  # Override in specific tests
    
    # History deals mock
    mock.history_deals_get.return_value = []
    
    import sys
    sys.modules["MetaTrader5"] = mock
    # Also patch in each module that imports it
    monkeypatch.setattr("core.market.mt5", mock)
    monkeypatch.setattr("core.position.mt5", mock)
    return mock


@pytest.fixture
def sample_candles():
    """Generate sample OHLCV candle data for testing."""
    n = 100
    base_price = 82000.0
    times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h")
    
    # Create realistic price movement
    closes = []
    for i in range(n):
        if i == 0:
            closes.append(base_price)
        else:
            # Random walk with slight upward bias
            change = (i % 7 - 3) * 50 + (i % 13 - 6) * 20
            closes.append(closes[-1] + change)
    
    opens = [c - 10 + (i % 5) * 5 for i, c in enumerate(closes)]
    highs = [max(o, c) + 20 + (i % 3) * 10 for i, (o, c) in enumerate(zip(opens, closes))]
    lows = [min(o, c) - 20 - (i % 3) * 10 for i, (o, c) in enumerate(zip(opens, closes))]
    volumes = [100 + (i % 10) * 50 for i in range(n)]
    
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": volumes
    })


@pytest.fixture
def crossover_candles():
    """Generate candle data that creates EMA crossover signals.
    
    Crossover happens at completed bar (-2), with a forming candle at -1.
    """
    n = 101  # Extra bar for the "forming" candle
    times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h")
    
    # Create price series that will generate bullish crossover
    closes = []
    base = 82000.0
    
    for i in range(n):
        if i < 70:
            # Long downtrend to get fast below slow
            closes.append(base - i * 30)
        elif i < 85:
            # Consolidation
            closes.append(base - 70 * 30)
        elif i < n - 1:
            # Sharp uptrend for crossover
            closes.append(base - 70 * 30 + (i - 85) * 200)
        else:
            # Forming candle — same level as last completed
            closes.append(closes[-1])
    
    opens = [c - 5 for c in closes]
    highs = [max(o, c) + 10 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 10 for o, c in zip(opens, closes)]
    volumes = [100] * n
    
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": volumes
    })


@pytest.fixture
def temp_state_file():
    """Create a temporary state file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "daily_trade_count": 0,
            "weekly_trade_count": 0,
            "last_trade_ts": None,
            "last_candle_ts": None,
            "myt_date": None,
            "myt_week_iso": None,
            "tracked_positions": {}
        }, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_config():
    """Standard config for testing."""
    return {
        "symbol": "BTCUSD",
        "timeframe": "H1",
        "magic_number": 200001,
        "ema_fast": 10,
        "ema_slow": 50,
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.0,
        "risk_per_trade": 0.5,
        "min_lot": 0.01,
        "max_lot": 1.0,
        "max_open_positions": 3,
        "max_daily_trades": 50,
        "max_weekly_trades": 200,
        "cooldown_seconds": 300,
        "telegram_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "log_level": "INFO"
    }


@pytest.fixture
def mock_position():
    """Mock MT5 position object."""
    pos = Mock()
    pos.ticket = 123456
    pos.type = 0  # BUY
    pos.volume = 0.1
    pos.price_open = 82000.0
    pos.sl = 81800.0
    pos.tp = 82300.0
    pos.profit = 50.0
    pos.time = int(datetime.now(timezone.utc).timestamp())
    pos.magic = 200001
    return pos


@pytest.fixture
def mock_telegram():
    """Mock Telegram notification service."""
    telegram = Mock()
    telegram.enabled = False
    telegram.send.return_value = True
    return telegram


@pytest.fixture
def mock_state(temp_state_file):
    """Mock state with temporary file."""
    from utils.state import State
    return State(temp_state_file)


@pytest.fixture
def mock_market(mock_mt5, mock_config):
    """Mock market with MT5 mocked."""
    from core.market import Market
    return Market(mock_config)