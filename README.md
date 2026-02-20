# MT5 Trading Bot

Automated trading bots for MetaTrader 5 with machine learning signal prediction. BTCUSD bot runs live on a funded Pepperstone account. Demo bots (XAUUSD, EURUSD) currently paused due to MT5 multi-terminal limitations.

**Live Dashboard:** https://trade-bot-hq.vercel.app
**MQL5 Signal:** https://www.mql5.com/en/signals/2359964

## Live Trading (Feb 20, 2026)

| Metric | Value |
|--------|-------|
| **Live Account** | Pepperstone 51439249 (Razor, MT5) |
| **Live Balance** | ~$181 |
| **Live Bot** | BTCUSD (`btcusd-live/`) |
| **Total Trades** | 172 |
| Demo Balance | ~$49,577 |
| Demo P/L | +$1,178 |

### What's Running
- ✅ **bot-btcusd-live** — LIVE on account 51439249 (strict filters: 3/3 unanimous, 15 trades/week max)
- ✅ **mt5-watchdog** — Auto-restarts MT5 terminal if it crashes, sends Telegram alert
- ⏸️ mission-control — STOPPED (Node.js v24 compatibility issue)
- ⏸️ bot-btcusd (demo) — STOPPED
- ⏸️ bot-xauusd (demo) — STOPPED
- ⏸️ bot-eurusd (demo) — STOPPED

> **Why demo bots are stopped:** MT5 Python library only supports one terminal per process. Running multiple bots causes them to fight over MT5 connections, which caused a zombie XAUUSD bot to accidentally trade on the live account. Demo bots need a separate machine/VM.

### MQL5 Signal Provider
- **Signal:** [BTCUSD ML Scalpers](https://www.mql5.com/en/signals/2359964)
- **Status:** LIVE & APPROVED ✅
- **Price:** $30/month

## BTCUSD Live Bot — Trading Logic (v2, Feb 20)

### Signal Generation
1. Fetch 100 M5 candles + H1 candles from MT5
2. Engineer **28 features** (RSI, MACD, ATR, Bollinger, EMA, momentum, volume, candle patterns)
3. Run **3 ML models**: Random Forest, XGBoost, LightGBM
4. **Unanimous 3/3 vote required** — all models must agree on BUY or SELL
5. Average confidence must be ≥ 55% (session-adjusted)

### Filter Chain (any filter blocks the trade)
| Filter | Rule |
|--------|------|
| **EMA Trend** | BUY only in uptrend (EMA50 > EMA200), SELL only in downtrend |
| **Momentum** | Price must move ≥0.1% in signal direction over last 3 candles |
| **ATR Floor** | ATR(14) must be ≥ 50 (filters dead/chop zones) |
| **Spread** | Max 0.05% of price |
| **Off-Hours** | 00:00-06:00 MYT requires 75% confidence (base 55% + 20%) |
| **Trade Cooldown** | Min 180s between trades |
| **Circuit Breaker** | 3 consecutive losses = shutdown for rest of MYT day |
| **Weekly Limit** | Max 15 trades per MYT week (Mon-Sun) |

### Execution
- **Lot:** Fixed 0.01 (hard cap for $200 account)
- **Dynamic SL/TP:** SL = 1.0× ATR, TP = 1.5× ATR (scales with volatility)
- **Trailing stop:** Enabled
- **Partial profit:** At 1R, close 50% + move SL to breakeven
- **Min hold floor:** 15 minutes before trailing stop or stagnant exit can trigger
- **Max hold:** 120 minutes

### Observability
- **blocked_signals.csv** — Logs every blocked signal with reason, model votes, ATR, momentum
- **trade_log.csv** — All executed trades with P/L
- **Telegram notifications** — Trade entries, exits, errors, daily digest

## Supported Pairs & ML Strategy

| Bot | Symbol | ML Strategy | Confidence | Key Features |
|-----|--------|------------|------------|--------------|
| BTCUSD Live | Bitcoin/USD | **Ensemble** (RF + XGB + LGB) | 55% (75% off-hours) | 3/3 unanimous vote, momentum filter, dynamic SL/TP, weekly limit |
| XAUUSD | Gold/USD | XGBoost | 50% (55% Asian) | Reversal confirmation, crash detector, partial profit at 1R |
| EURUSD | Euro/USD | XGBoost | 40% (50% Asian session) | Tight scalping (15 pip SL, 20 pip TP), M5 timeframe |

### ML Pipeline
- **28 features** per prediction: RSI, MACD, Bollinger Bands, EMA crossovers, ATR, volume ratio, momentum, crash detection metrics, and more
- **3-class prediction:** BUY / SELL / HOLD with probability scores
- **BTCUSD Ensemble:** Three models vote independently. Signal only fires when **3/3 agree unanimously**
- **Auto-retrain:** Weekly automated retraining (Sunday 3 AM MYT) with accuracy validation, backup/rollback safety, and PM2 auto-restart
- **Class weighting:** SELL=2x, BUY=1x, HOLD=0.5x to counteract bullish bias in training data

## Project Structure

```
mt5-trading/
├── btcusd-live/         # LIVE Bitcoin bot (strict filters)
│   ├── trading.py       # Live bot logic (~2000 lines)
│   ├── config.json      # Conservative: 0.01 lot, 3/3 vote, 15/week
│   ├── blocked_signals.csv  # Every blocked signal with reason
│   └── ...
├── btcusd/              # Bitcoin bot + ML pipeline (DEMO - stopped)
│   ├── trading.py       # Main bot logic
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
├── gen_analysis.py      # Generate daily AI analysis
├── save_daily_analysis.py # Save daily AI analysis to Supabase
└── sync_to_supabase.py  # Incremental trade sync to cloud
```

## Features

### Trading
- **ML-Powered Signals:** Trained models predict BUY/SELL/HOLD with confidence scores
- **Ensemble Voting (BTCUSD):** 3/3 unanimous agreement required — highest conviction trades only
- **EMA Trend Filter:** Only BUY in uptrend, only SELL in downtrend
- **Momentum Pre-Check:** Signal must align with recent price direction (0.1% over 3 candles)
- **Dynamic SL/TP:** Scales with ATR — wider in volatility, tighter in calm markets
- **Session-Aware Trading:** Off-hours (00:00-06:00 MYT) requires 75% confidence
- **Partial Profit Taking:** Close 50% at 1:1 RR, move SL to breakeven
- **Min Hold Floor:** 15 minutes before any exit can trigger (let trades develop)

### Risk Management
- **Daily Circuit Breaker:** 3 consecutive losses = trading stops for the rest of the MYT day
- **Weekly Trade Limit:** Max 15 trades per week (Mon-Sun MYT) — forces quality over quantity
- **ATR Floor Filter:** Skips trades when ATR < 50 (low volatility chop)
- **Volatility Filter:** Skips trades when ATR > 2x rolling average
- **Adaptive Cooldown:** 5min base, +5min per consecutive loss (max 30min)
- **Crash Detector:** Halts trading 30min if price moves >3% in 15 minutes
- **Spread Filter:** Skips trades when spread exceeds 0.05% of price

### Observability
- **Blocked Signals Log:** Every blocked signal saved to CSV with reason, model votes, confidence, ATR
- **Cloud Dashboard:** Real-time monitoring at https://trade-bot-hq.vercel.app
- **Performance Metrics:** Sharpe ratio, max drawdown, win streaks, profit factor, equity curve
- **Telegram Notifications:** Trade entries, exits, balance updates, daily digest
- **Supabase Sync:** Real-time trade data + 30-min incremental safety sync

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

# Start live bot
pm2 start ecosystem.config.js --only bot-btcusd-live
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

# Auto-retrain all bots
python auto_retrain.py              # Retrain if model > 7 days old
python auto_retrain.py --force      # Force retrain now
python auto_retrain.py --dry-run    # Preview without changes
python auto_retrain.py --bot btcusd # Specific bot only
```

## Configuration

### BTCUSD Live (`btcusd-live/config.json`)
| Setting | Value |
|---------|-------|
| ML Strategy | Ensemble (RF+XGB+LGB), 3/3 unanimous |
| Confidence | 55% base, 60% Asian, 75% off-hours |
| Lot Size | 0.01 (fixed) |
| Dynamic SL | 1.0× ATR |
| Dynamic TP | 1.5× ATR |
| Min Hold | 15 minutes |
| Circuit Breaker | 3 losses = daily shutdown |
| Weekly Limit | 15 trades |
| EMA Filter | ✅ |
| Momentum Filter | ✅ (0.1%, 3 candles) |
| ATR Floor | 50 |
| News Filter | Disabled (bug — to be fixed) |

## PM2 Commands

```bash
pm2 status                    # Check all bots
pm2 logs bot-btcusd-live      # View live bot logs
pm2 restart bot-btcusd-live   # Restart live bot
pm2 stop all                  # Stop everything
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
