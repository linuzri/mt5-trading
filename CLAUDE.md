# CLAUDE.md - AI Agent Context

This file provides context for AI agents working on this codebase.

## Project Overview

Automated MT5 trading bot with ML-based signal prediction. BTCUSD bot with ensemble ML (RF + XGBoost + LightGBM), binary classification, H1 timeframe.

### Current Status (Feb 23, 2026)
- **LIVE:** Account 51439249 — **STOPPED** (pending demo validation of new H1 model)
- **DEMO:** Account 61459537 — Running from `btcusd/` (PM2 process: `bot-btcusd`)
- **Demo Week:** Feb 24-28 — clean evaluation, partial profit DISABLED, state persistence ACTIVE
- **Model:** H1 binary classification, 63.6% walk-forward accuracy, 1.93 profit factor in backtest
- **MQL5 Signal:** https://www.mql5.com/en/signals/2359964 — LIVE, APPROVED ✅
- **Auto-retrain:** Weekly Sunday 3AM MYT via `auto_retrain.py` cron
- **Auto-merge PRs:** Granted — merge directly without review

### Feb 23 Fixes (Critical)
- **State persistence (`state.json`):** `daily_trade_count`, `weekly_trade_count`, `consecutive_losses`, `circuit_breaker_triggered` now saved after every trade close and loaded on startup. No more counter resets on PM2 restart.
- **Sticky circuit breaker:** Once triggered (5 consecutive losses), wins do NOT reset `consecutive_losses`. Stays active until midnight MYT.
- **Partial profit DISABLED:** For clean demo evaluation. Was creating $0 P/L trades (breakeven SL too tight for H1 ATR).
- **Directory migration:** Demo bot moved from `btcusd-live/` to `btcusd/`. Live codebase untouched for isolation.
- **Supabase key updated:** New `sb_secret_` format key (old JWT keys deprecated).

### The H1 Breakthrough (Feb 22, 2026)
Previous M5 model had 37.6% win rate, 93% SHORT bias, -5.18% growth. After systematic experimentation:

| Experiment | Result | Deployed? |
|-----------|--------|-----------|
| 3-class, M5, 180d, 28 features | 36.4% accuracy | ❌ |
| Binary, M5, 15 features | 51.8% walk-forward, 26% SELL recall | ❌ |
| **Binary, H1, 16 features, balanced downsampling** | **63.6% walk-forward, 65.25% test** | ✅ Demo |

Key changes that worked:
- **H1 timeframe** (was M5) — spread is 6% of TP instead of 15%
- **Binary classification** — drop HOLD rows, BUY vs SELL only (baseline 50%)
- **Training data downsampling** — equal BUY/SELL counts per model (no class weights)
- **16 features** — `daily_range_position` is #1 most important feature
- **365 days training** — covers multiple market regimes
- **TP=0.5%/SL=0.4%** ($500/$400 at BTC $100K), 12-candle lookahead (12 hours)

Backtest results (0.01 lots, $200 account):
- 64.1% WR, 1.93 profit factor, 4.76 Sharpe
- Max DD: 9.8% ($25.76), equity never below $198.49
- +$363 total P/L (181% return), 312 trades over ~110 days

### Go-Live Criteria (after 15-20 demo trades)
- Win rate > 55%
- Profit factor > 1.3
- BUY/SELL split 35-65%
- No single-day drawdown > 5%
- **Nazri must give explicit approval**

## Architecture

```
btcusd-live/trading.py ─→ MetaTrader 5 API ─→ Pepperstone (demo: 61459537)
         │
         ├─→ logs/signals.csv (every ML signal, executed + blocked)
         ├─→ logs/trades.csv (every closed trade with full details)
         ├─→ logs/daily_summary.csv (daily rollup metrics)
         ├─→ blocked_signals.csv (legacy blocked signal log)
         ├─→ trade_notifications.log (verbose activity log)
         ├─→ demo_logger.py (structured CSV logging module)
         ├─→ Telegram notifications
         └─→ sync_to_supabase.py ─→ Supabase (cloud DB)
                                          │
              vercel-dashboard/ ─→ Cloud UI (Vercel)
```

## Bot Loop (H1 Binary Classification)

1. Every 60 seconds: fetch 100 H1 candles from MT5
2. Calculate **16 features** (bb_upper, bb_lower, range_position, price_vs_ema200, atr_14, drawdown_pct, bb_width, trend_strength, macd_signal, volume_ratio, price_vs_ema50, macd_line, rsi_14, price_vs_ema20, hourly_return, daily_range_position)
3. Run ensemble prediction: RF, XGB, LGB each vote **BUY or SELL** (binary, no HOLD class)
4. **2/3 majority agreement** — if models disagree, no trade (HOLD = disagreement)
5. Apply filter chain (in order):
   a. Confidence gate (≥60%, session-adjusted: +10% off-hours)
   b. Min probability diff (≥15% between BUY and SELL)
   c. ATR floor (≥50)
   d. Spread filter (≤0.05% of price)
   e. Trade cooldown (3600s = 1 hour between trades)
   f. Circuit breaker (5 consecutive losses = daily shutdown)
   g. Daily limit (3 trades/day max)
   h. Weekly limit (15 trades/week)
6. If signal passes ALL filters → execute with dynamic SL/TP (1.0×/1.5× ATR)
7. Monitor open positions: trailing stop, partial profit at 1R (after 15min hold floor)
8. Log to `logs/signals.csv` (all signals) and `logs/trades.csv` (closed trades)

## ML Pipeline Details

### Binary Classification
- **Classes:** SELL (0), BUY (1) — no HOLD class
- **Label method:** `sltp_aware` — BUY if long TP hits before SL, SELL if short TP hits before SL
- **HOLD rows dropped** from training (~16.5% of data = untradeable sideways candles)
- **Spread cost** (0.03%) baked into TP calculation during labeling

### Training Data Balancing
- Each model (RF, XGB, LGB) trains on **downsampled** data with equal BUY/SELL counts
- `_balance_training_data()` randomly samples minority class count from majority
- No class_weight parameters — let balanced data speak

### Walk-Forward Validation
- 5 splits, 6-month train window, 1-month test window
- Auto-detects candles/day from data density
- Reports mean accuracy ± std, min/max, break-even threshold (0.48)

### Feature Importance Analysis
- Prints top 15 + bottom features after training
- `daily_range_position` consistently #1 (where price sits in today's high-low range)

## Key Files

### Live Bot (currently stopped)
- `btcusd-live/trading.py` — Main bot loop (~2200 lines). ALL trading logic.
- `btcusd-live/config.json` — Runtime config (H1, 0.01 lots, 3600s cooldown, 5 max consec losses)
- `btcusd-live/ml_config.json` — ML config (H1, 365d, binary, 16 features, 60% confidence)
- `btcusd-live/ml/ensemble_predictor.py` — Ensemble voting (2/3 majority, binary)
- `btcusd-live/ml/ensemble_trainer.py` — Training with balanced downsampling + walk-forward
- `btcusd-live/ml/feature_engineering.py` — 16 features including hourly_return, daily_range_position
- `btcusd-live/demo_logger.py` — Structured CSV logging for demo validation
- `btcusd-live/backtest_ml.py` — ML-aware backtester with realistic position sizing
- `btcusd-live/logs/` — trades.csv, signals.csv, daily_summary.csv
- `btcusd-live/mt5_auth_live.json.bak` — LIVE credentials backup (DO NOT DELETE)

### Infrastructure
- `ecosystem.config.js` — PM2 config
- `auto_retrain.py` — Weekly model retraining
- `sync_to_supabase.py` — Trade data sync
- `daily_digest.py` — End-of-day summary

## Important Patterns

- **Binary Ensemble Voting:** `ensemble_predictor.py` — `most_common_count >= 2` for majority. HOLD = models disagree, not a predicted class.
- **Balanced Downsampling:** `_balance_training_data()` in `ensemble_trainer.py` — called before each model fit.
- **Session Trading:** Asian, EU, US sessions. Off-hours = +10% confidence threshold.
- **Circuit Breaker:** 5 consecutive losses → daily shutdown. STICKY: wins don't reset once triggered. $0.50 min win to reset counter (only if breaker hasn't fired).
- **State Persistence:** `save_state()` writes `state.json` after every trade close. `load_state()` on startup restores counters. Cross-checked with CSV restore (takes max). Resets at MYT midnight (daily) and Monday 00:00 MYT (weekly).
- **Dynamic SL/TP:** SL = 1.0× ATR, TP = 1.5× ATR.
- **Demo Logger:** `demo_logger.py` hooks into `log_blocked_signal()` and `log_trade_result()`.
- **Walk-Forward:** `walk_forward_validate()` in ensemble_trainer — 5 splits, rolling train/test.

## Config Quick Reference

### `btcusd-live/config.json`
| Setting | Value |
|---------|-------|
| Timeframe | H1 |
| Lot Size | 0.01 |
| Trade Cooldown | 3600s (1 hour) |
| Circuit Breaker | 5 consecutive losses |
| Weekly Limit | 15 trades |
| EMA Trend Filter | Disabled |
| Momentum Filter | Disabled |

### `btcusd-live/ml_config.json`
| Setting | Value |
|---------|-------|
| Timeframe | H1 |
| Training Period | 365 days |
| Lookahead | 12 candles (12 hours) |
| TP / SL | 0.5% / 0.4% |
| Confidence | 60% |
| Min Prob Diff | 15% |
| Max Trades/Day | 3 |
| Features | 16 (binary, no HOLD class) |

## Development Guidelines

- **Branch:** `main` only. Auto-merge PRs granted.
- **Testing:** Restart PM2 after code changes: `pm2 restart bot-btcusd --update-env`
- **SECURITY:** Never commit tokens/keys. Telegram token in env var `TELEGRAM_BOT_TOKEN`.
- **Windows:** PowerShell syntax. ASCII-safe print (no emoji — cp1252 crashes).
- **Demo vs Live:** Demo runs from `btcusd/`, live codebase in `btcusd-live/` (untouched). `mt5_auth.json` controls which account. Live backup at `btcusd-live/mt5_auth_live.json.bak`.
- **Directory isolation:** ALL demo changes go in `btcusd/`. NEVER modify `btcusd-live/` during demo week.
