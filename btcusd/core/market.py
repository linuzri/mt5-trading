"""MT5 connection, price/candle data, account info."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


class Market:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.symbol = config["symbol"]
        self.tf = TIMEFRAME_MAP[config.get("timeframe", "H1").upper()]
        self._connected = False
        self._auth = self._load_auth()

    def _load_auth(self) -> dict:
        auth_path = Path("mt5_auth.json")
        if not auth_path.exists():
            log.warning("mt5_auth.json not found — will attempt connectionless init")
            return {}
        with open(auth_path) as f:
            return json.load(f)

    def connect(self) -> bool:
        """Initialize MT5 terminal. Returns True if connected."""
        if mt5.terminal_info() is not None:
            self._connected = True
            return True

        mt5_path = self.config.get("mt5_path", "")
        auth = self._auth

        kwargs = {}
        if mt5_path:
            kwargs["path"] = mt5_path
        if auth.get("login"):
            kwargs["login"] = int(auth["login"])
        if auth.get("password"):
            kwargs["password"] = auth["password"]
        if auth.get("server"):
            kwargs["server"] = auth["server"]

        if not mt5.initialize(**kwargs):
            log.error("MT5 initialize() failed: %s", mt5.last_error())
            return False

        if not mt5.symbol_select(self.symbol, True):
            log.error("Failed to select symbol %s", self.symbol)
            mt5.shutdown()
            return False

        self._connected = True
        log.info("Connected to MT5. Account: %s", mt5.account_info().login if mt5.account_info() else "?")
        return True

    def ensure_connected(self) -> bool:
        if mt5.terminal_info() is None:
            self._connected = False
            return self.connect()
        return True

    def get_candles(self, count: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles. Returns DataFrame with columns [time, open, high, low, close, volume]."""
        if not self.ensure_connected():
            return None
        rates = mt5.copy_rates_from_pos(self.symbol, self.tf, 0, count)
        if rates is None or len(rates) == 0:
            log.warning("copy_rates_from_pos returned nothing for %s", self.symbol)
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def get_tick(self) -> Optional[tuple]:
        """Returns (bid, ask, spread_pct) or None."""
        if not self.ensure_connected():
            return None
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        spread_pct = (tick.ask - tick.bid) / tick.ask if tick.ask else 0.0
        return tick.bid, tick.ask, spread_pct

    def get_account(self) -> Optional[tuple]:
        """Returns (balance, equity) or None."""
        if not self.ensure_connected():
            return None
        acct = mt5.account_info()
        if acct is None:
            return None
        return acct.balance, acct.equity

    def get_positions(self) -> list:
        """Return open positions for this symbol filtered by magic number."""
        if not self.ensure_connected():
            return []
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        magic = self.config.get("magic_number", 200001)
        return [p for p in positions if p.magic == magic]

    def get_filling_mode(self) -> int:
        sym_info = mt5.symbol_info(self.symbol)
        if sym_info is None:
            return mt5.ORDER_FILLING_IOC
        if sym_info.filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        elif sym_info.filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def is_market_open(self) -> bool:
        if not self.ensure_connected():
            return False
        sym_info = mt5.symbol_info(self.symbol)
        if sym_info is None:
            return False
        return sym_info.visible and sym_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

    def shutdown(self) -> None:
        mt5.shutdown()
        self._connected = False
