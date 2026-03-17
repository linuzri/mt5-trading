"""Persist bot state across restarts."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_STATE = {
    "daily_trade_count": 0,
    "weekly_trade_count": 0,
    "last_trade_ts": None,
    "last_candle_ts": None,
    "myt_date": None,
    "myt_week_iso": None,
    "tracked_positions": {},  # ticket -> {direction, entry_price, open_time, lot, sl, tp}
}


class State:
    def __init__(self, path: str = "logs/state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
                # Ensure tracked_positions exists (upgrade from old state)
                if "tracked_positions" not in self._data:
                    self._data["tracked_positions"] = {}
                log.info("State loaded from %s", self.path)
            except Exception as e:
                log.warning("Failed to load state: %s — using defaults", e)
                self._data = dict(DEFAULT_STATE)
        else:
            self._data = dict(DEFAULT_STATE)

    def save(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            log.error("Failed to save state: %s", e)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value

    def reset_daily_if_needed(self, myt_date: str) -> None:
        if self._data.get("myt_date") != myt_date:
            self._data["daily_trade_count"] = 0
            self._data["myt_date"] = myt_date
            self.save()
            log.info("Daily counters reset for %s", myt_date)

    def reset_weekly_if_needed(self, myt_week_iso: str) -> None:
        if self._data.get("myt_week_iso") != myt_week_iso:
            self._data["weekly_trade_count"] = 0
            self._data["myt_week_iso"] = myt_week_iso
            self.save()
            log.info("Weekly counters reset for week %s", myt_week_iso)

    def record_trade(self) -> None:
        """Increment trade counters and set last trade timestamp."""
        self._data["daily_trade_count"] = self._data.get("daily_trade_count", 0) + 1
        self._data["weekly_trade_count"] = self._data.get("weekly_trade_count", 0) + 1
        self._data["last_trade_ts"] = datetime.now(timezone.utc).isoformat()
        self.save()

    # -- Position tracking (persisted) --

    def track_position(self, ticket: int, info: dict) -> None:
        """Save an open position to state for restart recovery."""
        self._data.setdefault("tracked_positions", {})
        self._data["tracked_positions"][str(ticket)] = info
        self.save()

    def untrack_position(self, ticket: int) -> dict:
        """Remove and return a tracked position."""
        positions = self._data.get("tracked_positions", {})
        info = positions.pop(str(ticket), None)
        if info is not None:
            self.save()
        return info

    def get_tracked_positions(self) -> dict:
        """Return all tracked positions {ticket_str: info}."""
        return dict(self._data.get("tracked_positions", {}))

    def as_dict(self) -> dict:
        return dict(self._data)
