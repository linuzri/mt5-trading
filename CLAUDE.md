# CLAUDE.md - AI Agent Context

This file provides context for AI agents working on this codebase.

## Project Overview

Automated MT5 BTCUSD trading bot — **v2 architecture** (March 14, 2026). Clean rewrite from scratch. Simple, modular, config-driven.

### Current Status (Mar 14, 2026)
- **DEMO:** Account 61459537 — Running from `btcusd/` (PM2: `bot-btcusd-v2`)
- **LIVE:** Account 51439249 (Pepperstone Razor) — **STOPPED** (no timeline, earn it on demo first)
- **Strategy:** `ema_cross` — EMA fast/slow crossover on H1
- **Philosophy:** Simple first, collect data, analyze, iterate. No time pressure.
- **Auto-merge PRs:** Granted — merge directly without review

### v1 Archive
- **Location:** `btcusd-v1-archive/` — full v1 codebase preserved for reference
- **Why archived:** 2500+ line monolith, 12+ stacked filters, untraceable trade decisions
- **AutoResearch (v1):** Stopped. Both param tuner and strategy discovery were v1-specific
- **MQL5 Signal:** https://www.mql5.com/en/signals/2359964 — still live but no active bot

## v2 Architecture

```
btcusd/
├── config.yaml              ← Single source of truth (all params)
├── main.py                  ← Entry point (~30 lines)
├── core/
│   ├── engine.py            ← Main loop: candle → signal → risk → execute
│   ├── market.py            ← MT5 connection, price data, account info
│   └── position.py          ← Open/close positions, SL/TP
├── strategy/
│   ├── base.py              ← Abstract strategy interface
│   └── ema_cross.py         ← EMA crossover (first strategy)
├── risk/
│   ├── sizing.py            ← Position sizing (% risk)
│   └── limits.py            ← Daily/weekly caps, cooldown
├── notifications/
│   └── telegram.py          ← Trade notifications
├── logs/
│   └── trades.jsonl         ← Unified structured log (ALL events)
└── utils/
    └── state.py             ← State persistence across restarts
```

### Design Principles
1. **YAML config, not code changes.** Strategy, params, risk — all in config.yaml
2. **Pluggable strategies.** Add new .py file in strategy/, change config
3. **Unified JSONL logging.** Every signal, skip, trade, close — one file, structured
4. **No filter spaghetti.** Risk rules are explicit functions returning (allow, reason)
5. **Fast loop.** Check candle → ask strategy → check risk → execute. Clean.

### EMA Crossover Strategy
- EMA fast crosses above EMA slow → BUY
- EMA fast crosses below EMA slow → SELL
- Only on crossover candle (not while already crossed)
- SL = ATR × multiplier, TP = ATR × multiplier
- No filters, no confirmations, no multi-timeframe. Just cross and go.

## Config Quick Reference (config.yaml)

| Setting | Value | Notes |
|---------|-------|-------|
| Symbol | BTCUSD | |
| Magic Number | 200001 | v2 identifier (v1 used 100xxx) |
| Strategy | ema_cross | Pluggable |
| Timeframe | H1 | Primary |
| EMA Fast | 10 | |
| EMA Slow | 50 | |
| Risk/Trade | 0.5% | Of balance |
| SL | 2.0× ATR | |
| TP | 3.0× ATR | 1.5:1 R:R |
| Cooldown | 300s | Between trades |
| Daily Limit | 50 | Generous for data collection |
| Weekly Limit | 200 | Generous for data collection |
| Max Positions | 3 | Simultaneous |

## Key Files

| File | Purpose |
|------|---------|
| `btcusd/config.yaml` | All parameters — edit this, not code |
| `btcusd/logs/trades.jsonl` | Every decision logged — gold for analysis |
| `btcusd/main.py` | Entry point with --dry-run flag |
| `btcusd-v1-archive/` | Old bot, reference only |
| `ecosystem.config.js` | PM2 process management |

## JSONL Log Format

```json
{"ts": "...", "event": "signal", "direction": "buy", "reason": "EMA10 crossed above EMA50", ...}
{"ts": "...", "event": "risk_check", "allowed": true, "daily_trades": 3, ...}
{"ts": "...", "event": "trade_open", "ticket": 12345, "direction": "buy", "price": 71050, ...}
{"ts": "...", "event": "trade_close", "ticket": 12345, "profit": 150.50, "close_reason": "tp_hit"}
{"ts": "...", "event": "skip", "reason": "cooldown_active", "remaining_seconds": 120}
```

## Development Guidelines

- **Branch:** `main` only. Auto-merge PRs granted.
- **SECURITY:** NEVER commit tokens/keys/secrets. Use env vars or `.env` (gitignored).
- **Windows:** PowerShell syntax. ASCII-safe print (no emoji — cp1252 crashes).
- **Testing:** `python main.py --dry-run` for signal-only mode
- **Config changes:** Edit config.yaml only. Never hardcode params in .py files.
- **New strategies:** Create strategy/new_name.py implementing Strategy base class, set `strategy: new_name` in config.yaml

## Core Principles

- **Simplicity First:** If it's not in the spec, don't add it
- **Data Collection:** Log everything — we're learning, not optimizing yet
- **No Time Pressure:** Demo runs as long as needed. Quality over speed.
- **Iterate from data:** Collect → Analyze → Improve. Not guess → deploy → hope.
