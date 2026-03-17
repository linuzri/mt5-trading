# BTCUSD Trading Bot v2 — Build Task

## Goal
Build a clean, modular BTCUSD trading bot from scratch. Simple, config-driven, with unified logging for data collection and analysis.

## Architecture

```
btcusd/
├── config.yaml              ← All parameters, strategy selection, risk rules
├── main.py                  ← Entry point (~30 lines, just boot and run)
├── core/
│   ├── engine.py            ← Main loop: check candle → get signal → check risk → execute
│   ├── market.py            ← MT5 connection, price/candle data, account info
│   └── position.py          ← Open/close/manage positions, SL/TP calculation
├── strategy/
│   ├── base.py              ← Abstract base class for strategies
│   └── ema_cross.py         ← EMA crossover strategy (first strategy)
├── risk/
│   ├── sizing.py            ← Position sizing (% risk per trade)
│   └── limits.py            ← Daily/weekly trade caps, cooldown between trades
├── notifications/
│   └── telegram.py          ← Send trade notifications to Telegram
├── logs/
│   └── trades.jsonl         ← Unified structured log (auto-created)
└── utils/
    └── state.py             ← Persist bot state across restarts (JSON)
```

## Config (config.yaml)

```yaml
# Connection
symbol: "BTCUSD"
magic_number: 200001          # Unique ID for v2 trades (v1 used 100xxx)

# Strategy
strategy: "ema_cross"
timeframe: "H1"               # Primary timeframe

# EMA Crossover params
ema_fast: 10
ema_slow: 50

# Risk Management
risk_per_trade: 0.5            # % of balance per trade
sl_atr_multiplier: 2.0         # SL = ATR * this
tp_atr_multiplier: 3.0         # TP = ATR * this  (1.5:1 R:R)
atr_period: 14
min_lot: 0.01
max_lot: 1.0

# Limits
max_daily_trades: 50
max_weekly_trades: 200
cooldown_seconds: 300          # 5 min between trades
max_open_positions: 3          # Max simultaneous positions

# Notifications
telegram_enabled: true
telegram_bot_token: "${TELEGRAM_BOT_TOKEN}"   # From env var
telegram_chat_id: "${TELEGRAM_CHAT_ID}"       # From env var

# Logging
log_file: "logs/trades.jsonl"
log_level: "INFO"
```

## Strategy: EMA Crossover (ema_cross.py)

Dead simple:
1. Calculate EMA fast and EMA slow on H1 candles
2. When EMA fast crosses ABOVE EMA slow → BUY signal
3. When EMA fast crosses BELOW EMA slow → SELL signal
4. Only signal on the crossover candle (not while already crossed)
5. SL = entry price ∓ (ATR × sl_multiplier)
6. TP = entry price ± (ATR × tp_multiplier)

That's it. No filters, no confirmations, no multi-timeframe. Just cross and go.

## Strategy Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class Signal:
    direction: str           # "buy" or "sell"
    reason: str              # Human-readable reason
    sl_distance: float       # SL distance in price
    tp_distance: float       # TP distance in price
    confidence: float = 1.0  # 0-1, for future use

class Strategy(ABC):
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def evaluate(self, candles, config) -> Optional[Signal]:
        """Return Signal if entry condition met, None otherwise."""
        ...
```

## Engine Loop (core/engine.py)

```
while running:
    1. Get latest candles from MT5
    2. Check if new candle formed (dedup by candle timestamp)
    3. Ask strategy: signal = strategy.evaluate(candles, config)
    4. If signal:
       a. Check risk limits (daily cap, weekly cap, cooldown, max positions)
       b. If allowed: calculate lot size, open position
       c. Log everything (signal, decision, trade details)
    5. Check open positions for management (optional: trailing stop later)
    6. Sleep until next check (60 seconds)
```

## Unified Logging (JSONL)

Every event is one JSON line in trades.jsonl:

```json
{"ts": "2026-03-14T09:00:00Z", "event": "signal", "direction": "buy", "reason": "EMA10 crossed above EMA50", "ema_fast": 71000.5, "ema_slow": 70500.2, "atr": 450.0, "price": 71050.0}
{"ts": "2026-03-14T09:00:01Z", "event": "risk_check", "allowed": true, "daily_trades": 3, "weekly_trades": 15, "open_positions": 1}
{"ts": "2026-03-14T09:00:02Z", "event": "trade_open", "ticket": 12345, "direction": "buy", "price": 71050.0, "sl": 70150.0, "tp": 72400.0, "lot": 0.01, "reason": "EMA10 crossed above EMA50"}
{"ts": "2026-03-14T09:00:02Z", "event": "trade_close", "ticket": 12345, "profit": 150.50, "duration_hours": 4.2, "close_reason": "tp_hit"}
{"ts": "2026-03-14T09:00:00Z", "event": "skip", "reason": "cooldown_active", "remaining_seconds": 120}
```

This is gold for analysis. One file, structured, queryable.

## Key Differences from v1

| Aspect | v1 (Frankenstein) | v2 (Clean) |
|--------|-------------------|------------|
| Files | 1 file, 2500+ lines | 10+ files, <200 lines each |
| Config | JSON, mixed with code | YAML, single source of truth |
| Strategy | Hardcoded in trading.py | Pluggable modules |
| Filters | 12+ stacked, conflicting | Simple risk rules, explicit |
| Logging | Multiple files, inconsistent | Single JSONL, structured |
| New strategy | Edit trading.py (scary) | Add new .py file, change config |
| Tuning | Edit code or JSON | Edit config.yaml only |

## MT5 Connection Details

- **Broker:** Pepperstone
- **Account:** 61459537 (demo)
- **Symbol:** BTCUSD
- **Platform:** MetaTrader 5 (installed, working)
- **Python package:** MetaTrader5 (pip installed)

## Environment Variables (already set)

- TELEGRAM_BOT_TOKEN — for trade notifications
- TELEGRAM_CHAT_ID — Nazri's chat ID (3588682)
- ANTHROPIC_API_KEY — not needed for bot, but available

## Requirements

- Python 3.x with MetaTrader5, numpy, pandas, pyyaml
- No external APIs except MT5 and Telegram
- Must work on Windows (Nazri's laptop)
- PM2 managed (will add to ecosystem.config.js later)

## Important Notes

1. Use magic_number 200001 to avoid conflicting with any v1 leftover positions
2. Read Telegram credentials from environment variables, NEVER hardcode
3. The bot should log EVERYTHING — we're collecting data for analysis
4. Keep it simple. If you're adding a feature that isn't in this spec, DON'T.
5. Config should support env var interpolation for secrets (${VAR_NAME} syntax)
6. The v1 archive is at btcusd-v1-archive/ — you can reference it for MT5 connection patterns but DO NOT copy its architecture
