# MT5 Trading Bot — BTCUSD v2

Automated BTCUSD trading bot for MetaTrader 5. Simple, modular, config-driven.

## Quick Start

```bash
# Install dependencies
pip install MetaTrader5 numpy pandas pyyaml requests

# Dry run (logs signals without trading)
cd btcusd
python main.py --dry-run

# Live demo trading
cd btcusd
python main.py

# PM2 managed
pm2 start ecosystem.config.js --only bot-btcusd-v2
```

## Architecture

```
btcusd/
├── config.yaml         ← All parameters here
├── main.py             ← Entry point (~65 lines)
├── core/
│   ├── engine.py       ← Main loop, heartbeat, event logging
│   ├── market.py       ← MT5 connection, candles, ticks
│   └── position.py     ← Open/close/track positions, restart recovery
├── strategy/
│   ├── base.py         ← Strategy interface
│   └── ema_cross.py    ← EMA crossover implementation
├── risk/
│   ├── sizing.py       ← ATR-based position sizing
│   └── limits.py       ← Daily/weekly/max position limits
├── notifications/
│   └── telegram.py     ← Trade & heartbeat alerts
├── logs/
│   └── trades.jsonl    ← Unified structured event log
└── utils/
    └── state.py        ← Persistent state + position tracking
```

Every file is under 200 lines. No monoliths.

## Strategy: EMA Crossover

Dead simple:
- EMA(10) crosses above EMA(50) → **BUY**
- EMA(10) crosses below EMA(50) → **SELL**
- Timeframe: H1
- SL = ATR × 2.0, TP = ATR × 3.0 (1.5:1 R:R)
- Risk: 0.5% balance per trade

No filters. No multi-timeframe confirmation. Just cross and go.

## Configuration

All parameters live in `btcusd/config.yaml`. Edit config, restart bot. No code changes needed.

Key settings:
- `risk_pct: 0.5` — % of balance risked per trade
- `sl_atr_multiplier: 2.0` / `tp_atr_multiplier: 3.0` — stop/target distances
- `max_positions: 3` — concurrent position cap
- `max_daily_trades: 50` / `max_weekly_trades: 200` — rate limits
- `cooldown_seconds: 300` — minimum time between trades

## Logging

Every decision is logged to `logs/trades.jsonl` as structured JSONL:

| Event | Data |
|-------|------|
| `bot_start` | strategy, balance, equity, tracked positions |
| `bot_stop` | balance, equity at shutdown |
| `heartbeat` | balance, equity, open positions, unrealized PnL, session P/L |
| `signal` | direction, price, bid/ask, spread, EMA values, ATR |
| `risk_check` | allowed/blocked, daily/weekly counts, open positions |
| `trade_open` | ticket, direction, price, SL, TP, lot, spread, ATR, balance |
| `trade_close` | ticket, entry/close price, profit, pips, duration, close reason, balance after |
| `skip` | reason signal was blocked |
| `error` | exception details |

## Telegram Notifications

- **Trade open** — direction, price, SL/TP, lot, spread, balance
- **Trade close** — entry/exit, PnL, duration, close reason, balance
- **Hourly heartbeat** (at :00) — balance, equity, session P/L, open positions with live PnL, trade counts
- **Bot start/stop** — strategy, balance, mode

## Position Tracking

Positions are persisted to `logs/state.json` and survive bot restarts:
- On startup, scans MT5 for orphaned positions and recovers them
- Detects SL/TP closes with full deal history (close price, PnL, duration, reason)
- No more "ghost" positions lost on restart

## Adding a Strategy

1. Create `strategy/my_strategy.py` implementing the `Strategy` base class
2. Set `strategy: my_strategy` in `config.yaml`
3. Restart bot

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `TELEGRAM_BOT_TOKEN` | Trade notification bot token |
| `TELEGRAM_CHAT_ID` | Where to send notifications |

## History

- **v1** (Jan–Mar 2026): Monolithic 2500-line bot with ML ensemble, 12+ filters, multi-timeframe confirmation, AutoResearch optimizer (523 experiments, 70.6% WR). Over-engineered — too many filters meant too few trades. Archived in `btcusd-v1-archive/`.
- **v2** (Mar 14, 2026): Clean rewrite. Modular architecture, <200 lines per file, YAML config, unified JSONL logging, persistent position tracking, hourly Telegram heartbeats. Philosophy: simple strategy, rich data collection, iterate from evidence.

## Broker

- **Pepperstone** (demo account)
- Symbol: BTCUSD
- MetaTrader 5
- Magic number: 200001
