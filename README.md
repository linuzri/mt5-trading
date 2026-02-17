# MT5 Trading Bot

Automated trading bots for MetaTrader 5 with machine learning signal prediction. Three bots trade BTCUSD, XAUUSD, and EURUSD 24/7 on a demo account with real-time cloud monitoring.

**Live Dashboard:** https://trade-bot-hq.vercel.app

## Current Performance (Feb 17, 2026)

| Metric | Value |
|--------|-------|
| Account Balance | ~$49,577 |
| Grand Total P/L | **+$1,178** |
| BTCUSD (Star) | **+$220** |
| XAUUSD | **+$173** |
| EURUSD (Solid) | **+$155** |

### Recent Improvements (Feb 17)
- **XAUUSD reversal confirmation** — requires 2 consecutive signals before reversing direction (data: reversals 32.9% WR vs continuations 46.9% WR)
- **XAUUSD crash detector** — halts trading on >1.5% price move in 15min
- **Session-aware ATR for EURUSD** — per-session thresholds (Asian/EU/US) instead of one-size-fits-all
- **Unicode console fix** — safe ASCII encoding on all 3 bots for Windows compatibility
- **Symbol identifiers** — all notifications now prefixed with bot name (BTCUSD/XAUUSD/EURUSD)

## Supported Pairs & ML Strategy

| Bot | Symbol | ML Strategy | Confidence | Key Features |
|-----|--------|------------|------------|--------------|
| BTCUSD | Bitcoin/USD | **Ensemble** (Random Forest + XGBoost + LightGBM) | 55% (65% off-hours) | Majority vote (2/3 must agree), volatility filter, crash detector |
| XAUUSD | Gold/USD | XGBoost | 50% (55% Asian) | Reversal confirmation, crash detector, partial profit at 1R |
| EURUSD | Euro/USD | XGBoost | 40% (50% Asian session) | Tight scalping (15 pip SL, 20 pip TP), M5 timeframe |

### ML Pipeline
- **28 features** per prediction: RSI, MACD, Bollinger Bands, EMA crossovers, ATR, volume ratio, momentum, crash detection metrics, and more
- **3-class prediction:** BUY / SELL / HOLD with probability scores
- **BTCUSD Ensemble:** Three models vote independently. Signal only fires when 2/3 agree — filters out weak/conflicting signals for higher quality trades
- **Auto-retrain:** Weekly automated retraining (Sunday 3 AM MYT) with accuracy validation, backup/rollback safety, and PM2 auto-restart. Also runs on-demand — last run Feb 13 (ensemble accuracy 45.9%)
- **Class weighting:** SELL=2x, BUY=1x, HOLD=0.5x to counteract bullish bias in training data

## Project Structure

```
mt5-trading/
├── btcusd/              # Bitcoin bot + ML pipeline
│   ├── trading.py       # Main bot logic (~2000 lines)
│   ├── ml/              # ML modules
│   │   ├── ensemble_predictor.py  # Ensemble voting (RF+XGB+LGB)
│   │   ├── ensemble_trainer.py    # Train all 3 models
│   │   ├── model_predictor.py     # Single model prediction
│   │   ├── model_trainer.py       # Single model training
│   │   ├── feature_engineering.py # 28 technical features
│   │   └── data_preparation.py   # MT5 data fetching
│   ├── models/          # Trained models + scalers + backups
│   ├── config.json      # Bot runtime config
│   ├── ml_config.json   # ML training config
│   └── train_ml_model.py  # CLI: train models (--refresh, --ensemble)
├── xauusd/              # Gold bot (same structure, single XGBoost)
├── eurusd/              # Euro bot (same structure, single XGBoost)
├── dashboard/           # Local web dashboard (Flask)
├── vercel-dashboard/    # Cloud dashboard (Vercel + Supabase)
│   └── index.html       # Single-file dashboard with Chart.js
├── auto_retrain.py      # Weekly auto-retrain scheduler
├── ecosystem.config.js  # PM2 process manager config
├── daily_digest.py      # End-of-day performance summary
├── gen_analysis.py        # Generate daily AI analysis (fixes gap issue)
├── save_daily_analysis.py # Save daily AI analysis to Supabase
└── sync_to_supabase.py  # Incremental trade sync to cloud
```

## Features

### Trading
- **ML-Powered Signals:** Trained models predict BUY/SELL/HOLD with confidence scores
- **Ensemble Voting (BTCUSD):** 3 models must reach consensus — reduces false signals
- **EMA Trend Filter:** Only BUY in uptrend, only SELL in downtrend (BTCUSD, EURUSD)
- **Session-Aware Trading:** Adjusts confidence thresholds for Asian/EU/US sessions
- **Dynamic Position Sizing:** Risk-based lot calculation (0.5% per trade)
- **Partial Profit Taking:** Close 50% at 1:1 RR, move SL to breakeven
- **Smart Exit:** Closes stagnant trades after timeout (EURUSD: 60min)
- **Spread Filter:** Skips trades when spread exceeds threshold

### Risk Management
- **Volatility Filter (BTCUSD):** Skips trades when ATR > 2x rolling average
- **Adaptive Cooldown (BTCUSD):** 5min base, +5min per consecutive loss (max 30min)
- **Crash Detector (BTCUSD):** Halts trading 30min if price moves >3% in 15 minutes
- **Crash Detector (XAUUSD):** Halts trading 30min if price moves >1.5% in 15 minutes
- **Reversal Confirmation (XAUUSD):** Requires 2 consecutive signals before reversing direction — prevents whipsaw losses
- **Momentum Filter:** Continues profitable streaks, blocks losing directions
- **Session-Aware ATR (EURUSD):** Per-session thresholds (Asian=0.00003, EU/US=0.00008)

### Infrastructure
- **Auto-Retrain:** Weekly model retraining with accuracy validation and rollback safety
- **Cloud Dashboard:** Real-time monitoring at https://trade-bot-hq.vercel.app
- **Performance Metrics:** Sharpe ratio, max drawdown, win streaks, profit factor, equity curve
- **Supabase Sync:** Real-time trade data + 30-min incremental safety sync
- **Daily Analysis:** AI-generated trading analysis saved to Supabase

## Quick Start

### Prerequisites
- Python 3.10+
- MetaTrader 5 terminal running
- MT5 Python package (`pip install MetaTrader5`)
- PM2 (`npm install -g pm2`)

### Setup
```bash
# Install dependencies
pip install MetaTrader5 scikit-learn xgboost lightgbm pandas numpy joblib requests

# Configure MT5 credentials (create in each bot folder)
# mt5_auth.json: {"login": 12345, "password": "xxx", "server": "BrokerServer"}

# Start all bots
pm2 start ecosystem.config.js
pm2 save

# Check status
pm2 status
```

### Train ML Models
```bash
# Single model (XAUUSD/EURUSD)
cd xauusd && python train_ml_model.py --refresh

# Ensemble model (BTCUSD)
cd btcusd && python train_ml_model.py --refresh --ensemble

# Auto-retrain all bots (checks if due, validates accuracy)
python auto_retrain.py              # Retrain if model > 7 days old
python auto_retrain.py --force      # Force retrain now
python auto_retrain.py --dry-run    # Preview without changes
python auto_retrain.py --bot btcusd # Specific bot only
```

## Dashboard

**Cloud:** https://trade-bot-hq.vercel.app (Vercel + Supabase, auto-deploys from main)

**Sections:**
- 🤖 Bot Performance — Live status, today's P/L, trade count per bot
- 📊 Daily P/L Chart — Green/red bars with per-bot tooltip breakdown
- 📈 Performance Metrics — Sharpe ratio, max drawdown, win streaks, profit factor + equity curve
- 📊 Daily Analysis — AI-generated market analysis with date browser
- 📋 Trade History — Filterable trade log with stats
- 📝 Live Logs — Real-time bot activity feed

**Local:** Removed from PM2 (use Vercel dashboard only)

## Configuration

Each bot has `config.json` (runtime) and `ml_config.json` (ML training):

| Setting | BTCUSD | XAUUSD | EURUSD |
|---------|--------|--------|--------|
| ML Strategy | Ensemble (RF+XGB+LGB) | XGBoost | XGBoost |
| Confidence | 55% | 50% | 40% |
| Risk % | 0.5% | 0.5% | 0.5% |
| Max Lot | 0.05 | 0.05 | 0.05 |
| Cooldown | 5 min | 5 min | 10 min |
| EMA Filter | ✅ | ❌ | ✅ |
| Volatility Filter | ✅ | ❌ | ❌ |
| Crash Detector | ✅ (3%) | ✅ (1.5%) | ❌ |
| Reversal Confirm | ❌ | ✅ (2 signals) | ❌ |

## PM2 Commands

```bash
pm2 status                # Check all bots
pm2 logs bot-btcusd       # View BTCUSD logs
pm2 restart bot-xauusd    # Restart Gold bot
pm2 restart all           # Restart everything
pm2 stop all              # Stop everything
```

## Supabase Tables

| Table | Description |
|-------|-------------|
| `trades` | All closed trades (symbol, direction, entry/exit price, P/L, confidence) |
| `daily_pnl` | Daily aggregate P/L and trade stats |
| `daily_analysis` | AI-generated daily trading analysis |
| `bot_status` | Real-time bot heartbeat status |
| `account_snapshots` | Account balance history |
| `logs` | Bot activity logs |

## License

MIT
