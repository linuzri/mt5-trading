# CLAUDE.md - AI Agent Context

This file provides context for AI agents (Claude, etc.) working on this codebase.

## Project Overview

Automated MT5 trading bots with ML-based signal prediction. Three bots run simultaneously via PM2, each trading a different pair.

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

- **Momentum Filter:** Check `check_trend_momentum()` in trading.py. Uses last 3 trades to continue streaks or block losing directions.
- **Session Trading:** Asian (00:00-08:00 UTC), EU (08:00-14:00), US (14:00-21:00). Each has different confidence thresholds.
- **Cooldown:** Minimum wait between trades to prevent overtrading.
- **Circuit Breaker:** Currently disabled (set to 999) to allow demo learning.

## Development Guidelines

- **Branch:** `main` only. No other branches needed.
- **Testing:** Run on demo account first. Current demo balance ~$50,000.
- **Paths:** Bots reference `mt5_auth.json` in their own directory (gitignored).
- **Retraining:** `python train_ml_model.py --refresh` in each bot folder.
- **PM2:** Config in `ecosystem.config.js` at repo root. `pm2 restart all` after code changes.

## Cloud Infrastructure

- **Supabase:** Stores trades, bot status, daily P/L. Bots push data after each trade.
- **Vercel:** Static dashboard reading from Supabase REST API. Auto-deploys when `vercel-dashboard/` changes.
- **Vercel ignore step:** `git diff --quiet HEAD^ HEAD -- .` (skips deploy if dashboard unchanged)

## Common Issues

- **MT5 error 10018:** Market closed. Bot handles this with market hours check.
- **ML bias:** Models can learn directional bias from trending markets. Fix with class weighting (SELL=2x, BUY=1x, HOLD=0.5x).
- **Spread spikes:** During low liquidity, spreads widen. Spread filter catches this.
