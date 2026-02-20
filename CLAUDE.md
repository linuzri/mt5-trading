# CLAUDE.md - AI Agent Context

This file provides context for AI agents (Claude, etc.) working on this codebase.

## Project Overview

Automated MT5 trading bots with ML-based signal prediction. Three bots run simultaneously via PM2, each trading a different pair.

### Current Status (Feb 21, 2026)
- **LIVE:** Account 51439249 (Pepperstone Razor MT5) | Balance: ~$181 | Bot: `bot-btcusd-live` | 172 trades
- **Demo:** ~$49,577 | All-time P/L: +$1,178 | **ALL DEMO BOTS STOPPED** (MT5 conflict)
- **MQL5 Signal:** https://www.mql5.com/en/signals/2359964 — LIVE, APPROVED ✅
- **Mission Control:** STOPPED (Node.js v24 crash loop — backlogged)
- **MT5 Watchdog:** `mt5_watchdog.js` — auto-restarts MT5 if crashed, Telegram alerts
- **Auto-retrain:** Weekly Sunday 3AM MYT via `auto_retrain.py` cron
- **Auto-merge PRs:** Granted — merge directly without review

### Live Bot (`btcusd-live/`)
- **Directory:** `btcusd-live/` (separate from demo `btcusd/`)
- **PM2 Process:** `bot-btcusd-live`
- **Account:** 51439249 (Pepperstone-MT5-Live01, Razor, 1:500 leverage)
- **Lot size:** Fixed 0.01

### The Complete Overhaul (Week of Feb 17-21, 2026)
Starting point: **18.8% win rate, -$9.91 P/L.** Everything below was built this week:

| # | Change | Detail |
|---|--------|--------|
| ⏱️ | **15-min Hold Floor** | Trailing stop + stagnant exit cannot trigger until 15min old |
| 🛑 | **Daily Circuit Breaker** | 3 consecutive losses = shutdown for rest of MYT day (resets midnight) |
| 🔒 | **Circuit Breaker Loophole Closed** | $0.50 minimum profit to count as a "win" (no micro-win resets) |
| 🌙 | **Off-Hours Confidence 75%** | 00:00-06:00 MYT requires 75% (base 55% + 20%) |
| 📈 | **Momentum Pre-Check** | Signal must align with price direction (≥0.1% over 3 candles) |
| 📊 | **Dynamic SL/TP** | SL = 1.0× ATR, TP = 1.5× ATR (replaces fixed 200/300 pips) |
| 📋 | **Weekly Trade Cap** | Max 15 trades per MYT week (Mon-Sun) |
| 📝 | **Blocked Signals Logging** | Every blocked signal → `blocked_signals.csv` with full context |
| 🗳 | **3/3 Unanimous Voting** | All RF+XGB+LGB must agree (was 2/3 majority) |
| 🧠 | **Training Data 180 days** | Was 30 days — now uses full 6 months of MT5 M5 history |
| ⚖️ | **Balanced Class Weights** | Was SELL=2x, BUY=1x, HOLD=0.5x → now `'balanced'` (auto-adjusts) |
| 🏔️ | **ATR Floor Filter** | ATR(14) must be ≥ 50 (filters chop zones) |
| ⏳ | **General Trade Cooldown** | 180s between ALL trades |

Also: stagnant exit disabled (conflicts with trailing stop), news filter disabled (bug with `abs()`)

### ⚠️ Known Issues
- **News filter bug:** `abs()` in time calculation treated past events as upcoming. Disabled for now (news_block_minutes=0). Needs proper fix.
- **Weekly counter on first week:** Bot counted 172 pre-existing trades from CSV as this week's trades. Will reset next Monday.
- **Mission Control crash loop:** SyntaxError from Node.js v24 upgrade. PM2 process stopped. Low priority.
- **MT5 multi-terminal limitation:** Only one MT5 account per Python process. Demo bots need separate machine.

## Architecture

```
btcusd-live/trading.py ─→ MetaTrader 5 API ─→ Pepperstone (account 51439249)
         │
         ├─→ blocked_signals.csv (every blocked signal)
         ├─→ trade_log.csv (executed trades)
         ├─→ Telegram notifications
         └─→ sync_to_supabase.py ─→ Supabase (cloud DB)
                                          │
              vercel-dashboard/ ─→ Cloud UI (Vercel)
```

## Bot Loop (btcusd-live)

1. Every 60 seconds: fetch 100 M5 candles + H1 candles from MT5
2. Calculate 28 features (RSI, MACD, Bollinger, EMA, ATR, volume, momentum, etc.)
3. Run ensemble prediction: RF, XGB, LGB each vote BUY/SELL/HOLD
4. **3/3 unanimous agreement required** — if any model disagrees, no trade
5. Apply filter chain (in order):
   a. Confidence gate (≥55% avg, session-adjusted: +5% Asian, +20% off-hours)
   b. EMA trend filter (BUY only in uptrend, SELL only in downtrend)
   c. Momentum filter (price must move ≥0.1% in signal direction over 3 candles)
   d. ATR floor (≥50)
   e. Spread filter (≤0.05% of price)
   f. Trade cooldown (180s since last trade)
   g. Circuit breaker (3 consecutive losses = daily shutdown)
   h. Weekly limit (15 trades/week)
6. If signal passes ALL filters → execute with dynamic SL/TP (1.0×/1.5× ATR)
7. Monitor open positions: trailing stop, partial profit at 1R (after 15min hold floor)
8. Log blocked signals to `blocked_signals.csv`

## Key Files

### Live Bot
- `btcusd-live/trading.py` — Main bot loop (~2000 lines). ALL trading logic.
- `btcusd-live/config.json` — Runtime config (filters, thresholds, lot sizes)
- `btcusd-live/ml/ensemble_predictor.py` — Ensemble voting logic (3/3 unanimous)
- `btcusd-live/blocked_signals.csv` — Blocked signal audit log
- `btcusd-live/trade_log.csv` — Executed trades

### ML Pipeline
- `btcusd/ml/feature_engineering.py` — 28 feature calculation
- `btcusd/ml/ensemble_trainer.py` — Train RF+XGB+LGB
- `btcusd/train_ml_model.py` — CLI to retrain

### Infrastructure
- `ecosystem.config.js` — PM2 config (reads TELEGRAM_BOT_TOKEN from env)
- `auto_retrain.py` — Weekly model retraining
- `sync_to_supabase.py` — Trade data sync
- `daily_digest.py` — End-of-day summary

## Important Patterns

- **Ensemble Voting (BTCUSD):** `ml/ensemble_predictor.py` line ~202. `most_common_count >= 3` for unanimous.
- **Session Trading:** Asian (00:00-08:00 UTC), EU (08:00-14:00), US (14:00-21:00). Off-hours = everything else.
- **Circuit Breaker:** 3 consecutive losses → `circuit_breaker_triggered = True`, checks MYT date. Resets on new MYT day.
- **Weekly Limit:** Counts trades from CSV on startup. Increments on each trade. Resets when ISO week number changes.
- **Momentum Filter:** `(current_close - close_3_ago) / close_3_ago * 100`. BUY blocked if momentum < -0.1%, SELL blocked if > 0.1%.
- **Dynamic SL/TP:** `sl_points = int(atr * sl_atr_multiplier)`, `tp_points = int(atr * tp_atr_multiplier)`. Currently 1.0× and 1.5×.
- **Min Hold Floor:** 15 minutes. Check `minutes_held < min_hold_minutes` before any exit logic.
- **Blocked Signal Logging:** `log_blocked_signal()` writes to `blocked_signals.csv` with columns: timestamp, signal, reason, rf/xgb/lgb signals, confidence, atr, ema_trend, price_momentum.
- **EMA Trend Filter:** `ema50 > ema200` = UPTREND (only BUY), `ema50 < ema200` = DOWNTREND (only SELL).
- **Partial Profit:** At 1R (price moves SL distance in profit), close 50% + move SL to breakeven.
- **Adaptive Cooldown:** 5min base, +5min per consecutive loss (max 30min), resets on win.
- **Crash Detector:** Halts 30min if >3% move in 15min.

## Development Guidelines

- **Branch:** `main` only. No other branches needed.
- **Auto-merge:** PRs can be merged without review.
- **Testing:** Always restart PM2 after code changes: `pm2 restart bot-btcusd-live --update-env`
- **Paths:** Bots reference `mt5_auth.json` in their own directory (gitignored).
- **SECURITY:** Never commit tokens/keys. Use env vars or .env files. The Telegram token is in env var `TELEGRAM_BOT_TOKEN`.
- **Windows:** Use PowerShell syntax (semicolons not &&). ASCII-safe print (no emoji — cp1252 crashes).
- **Config changes:** Edit `config.json` then restart PM2. No code rebuild needed.

## Cloud Infrastructure

- **Supabase:** Tables: trades, daily_pnl, daily_analysis, bot_status, account_snapshots, logs
- **Vercel:** https://trade-bot-hq.vercel.app — auto-deploys when `vercel-dashboard/` changes
- **Daily Digest:** Cron at 11:55 PM MYT
- **Supabase Sync:** Every 30 min (incremental, no reimport)

## Common Issues

- **MT5 error 10018:** Market closed (weekends). Normal.
- **News filter stuck:** Known bug — disabled. Set `news_block_minutes: 0` in config.
- **Weekly limit on first run:** Old trades counted toward limit. Resets next Monday.
- **Balance shows $0.00 on startup:** MT5 not yet connected. Resolves after first cycle.
- **Mission Control crash:** Node.js v24 incompatibility. Low priority fix.
