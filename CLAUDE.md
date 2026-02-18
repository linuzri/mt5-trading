# CLAUDE.md - AI Agent Context

This file provides context for AI agents (Claude, etc.) working on this codebase.

## Project Overview

Automated MT5 trading bots with ML-based signal prediction. Three bots run simultaneously via PM2, each trading a different pair.

### Current Status (Feb 18, 2026)
- **LIVE:** Account 51439249 (Pepperstone Razor MT5) | Balance: ~$194 | Bot: `bot-btcusd-live`
- **Demo:** ~$49,577 | All-time P/L: +$1,178 | **ALL DEMO BOTS STOPPED** (MT5 conflict)
- **MQL5 Signal:** https://www.mql5.com/en/signals/2359955 — LIVE, $30/month, APPROVED ✅
- **Auto-retrain:** Weekly Sunday 3AM MYT via `auto_retrain.py` cron
- **Auto-merge PRs:** Granted — merge directly without review

### Live Bot (`btcusd-live/`)
- **Directory:** `btcusd-live/` (separate from demo `btcusd/`)
- **PM2 Process:** `bot-btcusd-live`
- **Account:** 51439249 (Pepperstone-MT5-Live01, Razor, 1:500 leverage)
- **First trade:** SELL @ 68199.73
- **Conservative settings:** max_lot 0.01, circuit_breaker 5 losses, cooldown 10min, confidence 60%
- **Notifications:** Prefixed `[LIVE]` in Telegram

### Recent Changes (Feb 18)
- **Live BTCUSD bot deployed** — `btcusd-live/` directory, PM2 process `bot-btcusd-live`
- **New Pepperstone account** — 51439249 replaced 51439211 (copy trading issues on old account)
- **Zombie XAUUSD bot killed** — PM2 showed stopped but process was alive, traded on live account
- **CSV restore bug fixed** — live bot was loading demo trade stats from CSV, caused false 9-loss cooldown
- **All demo bots stopped** — MT5 multi-terminal limitation (see below)

### ⚠️ MT5 Multi-Terminal Limitation
- MT5 Python library (`mt5.initialize()`) can only connect to ONE terminal per Python process
- Running multiple bots causes them to fight over the MT5 connection
- A "stopped" bot can still have a zombie process that trades on the wrong account
- **Current solution:** Only run live bot, all demo bots stopped
- **Future:** Need separate machine/VM for demo bots

### ⚠️ CSV Stats Restore Gotcha
- On startup, bots restore daily stats (win/loss count) from `trade_log.csv`
- If the live bot's CSV contains demo trades, it loads wrong stats
- This caused a false 9-consecutive-loss cooldown on the live bot
- **Fix:** Ensure each bot directory has its own clean CSV, never share across live/demo

## Architecture

```
btcusd/trading.py  ─┐
xauusd/trading.py  ─┼─→ MetaTrader 5 API ─→ Broker
eurusd/trading.py  ─┘
         │
         └─→ supabase_sync.py ─→ Supabase (cloud DB)
                                      │
         dashboard/server.py ─→ Local UI (Flask)
         vercel-dashboard/   ─→ Cloud UI (Vercel)
```

## Bot Loop (each bot)

1. Every 60 seconds: fetch latest candles from MT5
2. Calculate features (28 total: RSI, MACD, Bollinger, EMA, crash detection, etc.)
3. Run ML model prediction (Random Forest for BTC, XGBoost for XAU/EUR)
4. Apply filters: spread check, session confidence, momentum, market hours
5. If signal passes all filters with sufficient confidence → place trade
6. Monitor open positions, apply trailing stop / take profit

## Key Files Per Bot

- `trading.py` — Main bot loop (~2000 lines). All trading logic lives here.
- `config.json` — Runtime config (risk %, thresholds, lot sizes)
- `ml/feature_engineering.py` — Feature calculation (28 features)
- `ml/model_trainer.py` — Model training pipeline
- `ml/model_predictor.py` — Prediction wrapper
- `train_ml_model.py` — CLI to retrain model
- `supabase_sync.py` — Push trade data to cloud

## Important Patterns

- **Momentum Filter:** Check trend momentum section in trading.py. Uses last 3 trades to continue streaks or block losing directions. **Important:** If momentum flips signal but trend filter would block the flip, keeps original signal (prevents deadlock — fixed Feb 16).
- **Session Trading:** Asian (00:00-08:00 UTC), EU (08:00-14:00), US (14:00-21:00). Each has different confidence thresholds.
- **Cooldown:** Minimum wait between trades to prevent overtrading. BTCUSD: 5min, EURUSD: 10min.
- **Circuit Breaker:** Currently disabled (set to 999) to allow demo learning.
- **Volatility Filter (BTCUSD):** Skips trades when ATR > 2x rolling 20-period average.
- **Adaptive Cooldown (BTCUSD):** 5min base, +5min per consecutive loss (max 30min), resets on win.
- **Crash Detector (BTCUSD):** Halts trading 30min if price moves >3% in 15 minutes.
- **Crash Detector (XAUUSD):** Halts trading 30min if price moves >1.5% in 15 minutes.
- **Reversal Confirmation (XAUUSD):** Requires 2 consecutive signals before reversing direction. Variables: `reversal_pending_direction`, `reversal_signal_count`, `REVERSAL_CONFIRMATION_REQUIRED=2`. Prevents whipsaw losses.
- **Session-Aware ATR (EURUSD):** Per-session min ATR thresholds in config.json (Asian=0.00003, EU/US=0.00008). Global threshold was blocking all Asian trades.
- **EMA Trend Filter (BTCUSD, EURUSD):** Only BUY in uptrend, only SELL in downtrend.
- **Partial Profit Taking:** All bots close 50% at 1:1 RR, move SL to breakeven. Note: may not trigger on XAUUSD at minimum lot (0.05) due to lot size constraints.
- **Smart Exit (EURUSD):** Closes stagnant trades after 60min if price moved <0.003%.

## Development Guidelines

- **Branch:** `main` only. No other branches needed.
- **Testing:** Run on demo account first. Current demo balance ~$49,577. **Live bot in `btcusd-live/`.**
- **Paths:** Bots reference `mt5_auth.json` in their own directory (gitignored).
- **Retraining:** `python train_ml_model.py --refresh` in each bot folder.
- **Auto-Retrain:** `python auto_retrain.py` at repo root. Checks model age, retrains if > 7 days, validates accuracy (max 5% drop allowed), backs up old models, restarts PM2. Runs weekly via cron (Sunday 3 AM MYT). Use `--force` to retrain immediately, `--dry-run` to preview, `--bot <name>` for specific bot.
- **PM2:** Config in `ecosystem.config.js` at repo root. `pm2 restart all` after code changes.
- **SECURITY:** Never commit files with hardcoded tokens/keys. Use `.env` for secrets. Root-level `check_*.py` scripts are gitignored.

## Cloud Infrastructure

- **Supabase:** Stores trades, bot status, daily P/L, daily analysis. Bots push data after each trade.
- **Supabase Tables:** trades, daily_pnl, daily_analysis, bot_status, account_snapshots, logs
- **Supabase Management API:** Available for schema changes (token in .env as SUPABASE_MGMT_TOKEN)
- **Vercel:** Static dashboard reading from Supabase REST API. Auto-deploys when `vercel-dashboard/` changes.
- **Vercel ignore step:** `git diff --quiet HEAD^ HEAD -- .` (skips deploy if dashboard unchanged)
- **Daily Analysis:** `save_daily_analysis.py` writes to Supabase `daily_analysis` table + JSON backup. Dashboard reads from Supabase.
- **Daily Analysis Gap Fix:** `gen_analysis.py` generates analysis on-demand (fixes missed cron runs)

## Bot-Specific Settings

### BTCUSD
- Model: **Ensemble** (Random Forest + XGBoost + LightGBM) — majority vote, 2/3 must agree
- Confidence: 55% (65% off-hours) | EMA filter: ON
- Smart filters: volatility, adaptive cooldown, crash detector
- Ensemble predictor: `ml/ensemble_predictor.py` | Trainer: `ml/ensemble_trainer.py`
- Train: `python train_ml_model.py --refresh --ensemble`
- 24/7 trading (crypto never sleeps)

### XAUUSD  
- Model: XGBoost | SL: $40 | TP: $60
- **Reversal confirmation:** 2 consecutive signals required before reversing direction
- **Crash detector:** Halts trading on >1.5% price move in 15min
- Partial profit may not work at 0.05 lots (min lot constraint)
- Market hours: Mon-Fri only (closes Fri 22:00 UTC, opens Sun 22:00 UTC)

### EURUSD
- Model: XGBoost | Confidence: 40% (50% Asian session) | Timeframe: M5
- SL: 15 pips | TP: 20 pips (tightened for slow forex)
- EMA filter: ON | Cooldown: 10min | Smart exit: 60min max hold
- Market hours: Mon-Fri only

## Dashboard

- **Local:** Removed from PM2 (no longer used)
- **Cloud:** https://trade-bot-hq.vercel.app (Vercel + Supabase) — primary dashboard
- **Daily P/L chart:** Single green/red bars (total P/L). Hover tooltip shows per-bot breakdown (BTCUSD, XAUUSD, EURUSD).

## Common Issues

- **MT5 error 10018:** Market closed. Bot handles this with market hours check.
- **ML bias:** Models can learn directional bias from trending markets. Fix with class weighting (SELL=2x, BUY=1x, HOLD=0.5x).
- **Spread spikes:** During low liquidity, spreads widen. Spread filter catches this.
- **Supabase sync:** Real-time push works (bots push after each trade). Incremental sync cron runs every 30 min as safety net (only adds new trades, no longer clears+reimports).
- **Supabase schema changes:** Use Management API (`SUPABASE_MGMT_TOKEN` in .env) to run SQL. Direct psycopg2 connection doesn't work with Supabase pooler.
- **EURUSD stagnant trades:** Forex moves slowly. Smart exit closes dead trades after 60min.
