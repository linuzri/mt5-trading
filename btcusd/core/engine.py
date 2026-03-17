"""Main engine loop: candle check → signal → risk → execute → log."""
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.market import Market
from core.position import PositionManager
from notifications.telegram import Telegram
from risk.limits import Limits
from risk.sizing import Sizer
from strategy.base import Strategy
from utils.state import State

log = logging.getLogger(__name__)

MYT_OFFSET_HOURS = 8


def _now_myt() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=MYT_OFFSET_HOURS)


def _myt_date() -> str:
    return _now_myt().strftime("%Y-%m-%d")


def _myt_week_iso() -> str:
    d = _now_myt()
    return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"


class Engine:
    def __init__(
        self,
        config: dict,
        market: Market,
        strategy: Strategy,
        sizer: Sizer,
        limits: Limits,
        state: State,
        telegram: Telegram,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.market = market
        self.strategy = strategy
        self.sizer = sizer
        self.limits = limits
        self.state = state
        self.telegram = telegram
        self.dry_run = dry_run

        self.positions = PositionManager(market, config, state)
        self.log_path = Path(config.get("log_file", "logs/trades.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._running = True
        self._last_heartbeat_hour = None  # Track which hour we last sent
        self._session_trades_opened = 0
        self._session_trades_closed = 0
        self._session_pnl = 0.0
        self._session_start_balance = 0.0

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, *_) -> None:
        log.info("Shutdown signal received")
        self._running = False

    def _log_event(self, event: str, **kwargs) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
        line = json.dumps(record)
        log.info("EVENT %s", line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def run(self) -> None:
        log.info("Engine starting. Strategy=%s symbol=%s tf=%s dry_run=%s",
                 self.strategy.name, self.config["symbol"],
                 self.config.get("timeframe", "H1"), self.dry_run)

        if not self.dry_run:
            if not self.market.connect():
                log.error("Failed to connect to MT5 — exiting")
                return

        # Recover tracked positions from state + scan MT5
        if not self.dry_run:
            recovered = self.positions.recover_positions(self.market)
            if recovered:
                log.info("Recovered %d orphaned positions from MT5", recovered)

        # Startup log + notification
        acct_info = self.market.get_account() if not self.dry_run else None
        balance = acct_info[0] if acct_info else 0
        equity = acct_info[1] if acct_info else 0
        tracked_count = len(self.state.get_tracked_positions())

        self._log_event(
            "bot_start",
            strategy=self.strategy.name,
            symbol=self.config["symbol"],
            timeframe=self.config.get("timeframe", "H1"),
            balance=balance,
            equity=equity,
            tracked_positions=tracked_count,
            dry_run=self.dry_run,
        )

        self._session_start_balance = balance

        self.telegram.send(
            f"🤖 BTCUSD v2 bot started\n"
            f"Strategy: {self.strategy.name}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Open positions: {tracked_count}\n"
            f"Mode: {'DRY-RUN' if self.dry_run else 'LIVE DEMO'}"
        )

        poll_interval = 60

        while self._running:
            try:
                self._tick()
                self._maybe_heartbeat()
            except Exception as e:
                log.exception("Unhandled error in engine tick: %s", e)
                self._log_event("error", message=str(e))
            time.sleep(poll_interval)

        # Shutdown
        acct_info = self.market.get_account() if not self.dry_run else None
        self._log_event(
            "bot_stop",
            balance=acct_info[0] if acct_info else 0,
            equity=acct_info[1] if acct_info else 0,
            tracked_positions=len(self.state.get_tracked_positions()),
        )
        self.telegram.send("BTCUSD v2 bot stopped")

        log.info("Engine stopped")
        if not self.dry_run:
            self.market.shutdown()

    def _maybe_heartbeat(self) -> None:
        """Send hourly Telegram heartbeat at :00 with full status."""
        if self.dry_run:
            return

        now_myt = _now_myt()
        current_hour = now_myt.strftime("%Y-%m-%d-%H")

        # Only fire once per hour, and only after minute :00
        if current_hour == self._last_heartbeat_hour:
            return
        if now_myt.minute < 0:  # Always true, but keep for clarity
            pass

        self._last_heartbeat_hour = current_hour

        acct = self.market.get_account()
        balance = acct[0] if acct else 0
        equity = acct[1] if acct else 0
        positions = self.market.get_positions()
        open_count = len(positions)
        unrealized_pnl = sum(p.profit for p in positions) if positions else 0

        # Build position details
        pos_lines = []
        for p in (positions or []):
            direction = "🟢 BUY" if p.type == 0 else "🔴 SELL"
            pos_lines.append(
                f"  {direction} #{p.ticket} @ {p.price_open:.2f} "
                f"PnL: ${p.profit:+.2f}"
            )

        # Session stats
        session_pnl_change = balance - self._session_start_balance
        pnl_emoji = "📈" if session_pnl_change >= 0 else "📉"

        # Market status
        market_open = self.market.is_market_open()
        market_status = "🟢 Open" if market_open else "🔴 Closed"

        # Build message
        hour_str = now_myt.strftime("%I:%M %p")
        msg = (
            f"⏰ Hourly Report — {hour_str} MYT\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Balance: ${balance:,.2f}\n"
            f"💎 Equity: ${equity:,.2f}\n"
            f"{pnl_emoji} Session P/L: ${session_pnl_change:+,.2f}\n"
            f"📊 Market: {market_status}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📂 Open: {open_count}"
        )
        if unrealized_pnl != 0:
            msg += f" (${unrealized_pnl:+,.2f})"
        msg += "\n"

        if pos_lines:
            msg += "\n".join(pos_lines) + "\n"

        msg += (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Today: {self.state.get('daily_trade_count', 0)} trades | "
            f"Week: {self.state.get('weekly_trade_count', 0)} trades"
        )

        self.telegram.send(msg)

        # Also log to JSONL
        self._log_event(
            "heartbeat",
            balance=balance,
            equity=equity,
            open_positions=open_count,
            unrealized_pnl=round(unrealized_pnl, 2),
            tracked_count=len(self.state.get_tracked_positions()),
            session_pnl=round(session_pnl_change, 2),
            market_open=market_open,
        )

    def _tick(self) -> None:
        now_utc = datetime.now(timezone.utc)

        # Reset daily/weekly counters if needed
        self.state.reset_daily_if_needed(_myt_date())
        self.state.reset_weekly_if_needed(_myt_week_iso())

        # Check closed positions (SL/TP hit) — uses persisted tracking
        for closed in self.positions.check_closed_positions(self.market):
            self._log_event(
                "trade_close",
                ticket=closed["ticket"],
                direction=closed["direction"],
                entry_price=closed["entry_price"],
                close_price=closed["close_price"],
                profit=closed["profit"],
                profit_pips=closed["profit_pips"],
                lot=closed["lot"],
                sl=closed["sl"],
                tp=closed["tp"],
                duration_seconds=closed["duration_seconds"],
                close_reason=closed["close_reason"],
                balance_after=closed["balance_after"],
            )
            # Telegram close notification
            duration_h = closed["duration_seconds"] / 3600
            pnl_emoji = "✅" if closed["profit"] >= 0 else "❌"
            self.telegram.send(
                f"{pnl_emoji} CLOSED {closed['direction'].upper()} BTCUSD\n"
                f"Entry: {closed['entry_price']:.2f} → Exit: {closed['close_price']:.2f}\n"
                f"PnL: ${closed['profit']:+.2f} ({closed['close_reason']})\n"
                f"Duration: {duration_h:.1f}h | Balance: ${closed['balance_after']:,.2f}"
            )

        # Market open check
        if not self.dry_run and not self.market.is_market_open():
            log.debug("Market closed — skipping")
            return

        # Fetch candles
        candles = self.market.get_candles(count=200) if not self.dry_run else self._mock_candles()
        if candles is None or len(candles) < 2:
            log.warning("No candle data — skipping")
            return

        # Candle dedup
        last_candle_ts = str(candles["time"].iloc[-1])
        if last_candle_ts == self.state.get("last_candle_ts"):
            log.debug("Same candle — skipping")
            return
        self.state.set("last_candle_ts", last_candle_ts)
        self.state.save()

        # Strategy evaluation
        signal_obj = self.strategy.evaluate(candles)
        price = float(candles["close"].iloc[-1])

        if signal_obj is None:
            log.debug("No signal at %s price=%.2f", last_candle_ts, price)
            return

        # Get tick data for spread
        tick_data = self.market.get_tick() if not self.dry_run else (price, price + 5, 0.0001)
        bid = tick_data[0] if tick_data else price
        ask = tick_data[1] if tick_data else price
        spread = round(ask - bid, 2)

        self._log_event(
            "signal",
            direction=signal_obj.direction,
            reason=signal_obj.reason,
            price=price,
            bid=bid,
            ask=ask,
            spread=spread,
            **signal_obj.metadata,
        )

        # Risk check
        open_positions = self.market.get_positions() if not self.dry_run else []
        allowed, risk_reason = self.limits.check(self.state.as_dict(), len(open_positions))

        self._log_event(
            "risk_check",
            allowed=allowed,
            reason=risk_reason,
            daily_trades=self.state.get("daily_trade_count", 0),
            weekly_trades=self.state.get("weekly_trade_count", 0),
            open_positions=len(open_positions),
        )

        if not allowed:
            self._log_event("skip", reason=risk_reason)
            log.info("Signal blocked: %s", risk_reason)
            return

        # Position sizing
        acct = self.market.get_account() if not self.dry_run else (50000.0, 50000.0)
        balance = acct[0] if acct else 50000.0
        equity = acct[1] if acct else 50000.0
        lot = self.sizer.calculate_lot(balance, signal_obj.sl_distance)

        # SL / TP prices
        atr = signal_obj.sl_distance / self.config.get("sl_atr_multiplier", 2.0)
        sl, tp = self.positions.calculate_sl_tp(signal_obj.direction, price, atr)

        # Execute
        ticket = self.positions.open_position(
            direction=signal_obj.direction,
            lot=lot,
            sl=sl,
            tp=tp,
            comment=f"v2-{self.strategy.name}",
            dry_run=self.dry_run,
        )

        if ticket is not None:
            self.state.record_trade()
            self._log_event(
                "trade_open",
                ticket=ticket,
                direction=signal_obj.direction,
                price=price,
                sl=sl,
                tp=tp,
                lot=lot,
                spread=spread,
                atr=round(atr, 2),
                balance=balance,
                equity=equity,
                reason=signal_obj.reason,
                dry_run=self.dry_run,
            )
            msg = (
                f"{'[DRY-RUN] ' if self.dry_run else ''}"
                f"{'🟢 BUY' if signal_obj.direction == 'buy' else '🔴 SELL'} BTCUSD\n"
                f"Price: {price:.2f}  SL: {sl:.2f}  TP: {tp:.2f}\n"
                f"Lot: {lot}  Spread: {spread:.2f}\n"
                f"Reason: {signal_obj.reason}\n"
                f"Balance: ${balance:,.2f}"
            )
            self.telegram.send(msg)

    def _mock_candles(self):
        """Generate synthetic candles for dry-run testing."""
        import numpy as np
        import pandas as pd

        n = 200
        rng = np.random.default_rng(7)
        closes = np.full(n, 80000.0)
        closes[-1] = 82500.0

        times = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h", tz="UTC")
        noise = rng.normal(0, 10, n)
        opens = closes - np.abs(noise)
        highs = closes + rng.uniform(10, 80, n)
        lows = opens - rng.uniform(10, 80, n)
        vols = rng.integers(100, 1000, n)

        df = pd.DataFrame({
            "time": times, "open": opens, "high": highs,
            "low": lows, "close": closes, "tick_volume": vols,
        })
        df.iloc[-1, df.columns.get_loc("time")] = datetime.now(timezone.utc)
        return df
