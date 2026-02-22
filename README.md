# 🤖 MT5 Trading Bot — BTCUSD H1 ML Scalper

Automated Bitcoin trading bot for MetaTrader 5 with **ensemble ML (RF + XGBoost + LightGBM)**, binary classification, and H1 timeframe. Backtested at **64.1% win rate, 1.93 profit factor, 4.76 Sharpe ratio**.

**Live Dashboard:** https://trade-bot-hq.vercel.app  
**MQL5 Signal:** https://www.mql5.com/en/signals/2359964

## 🔴 Status (Feb 22, 2026)

| Item | Status |
|------|--------|
| **Live Bot** | ⏸️ STOPPED — pending demo validation |
| **Demo Bot** | ✅ Running (H1 binary model, account 61459537) |
| **Model** | 63.6% walk-forward accuracy, 65.25% test |
| **MQL5 Signal** | ✅ LIVE & APPROVED ($30/month) |
| **Auto-Retrain** | Weekly Sunday 3AM MYT |

### Backtest Results (0.01 lots, $200 account)
| Metric | Value |
|--------|-------|
| Win Rate | **64.1%** |
| Profit Factor | **1.93** |
| Sharpe Ratio | **4.76** |
| Max Drawdown | **9.8%** ($25.76) |
| Total P/L | **+$363** (181% return) |
| BUY / SELL WR | 61.7% / 66.9% (balanced ✅) |
| Trades | 312 over ~110 days |

## How It Works

```
Every 60 seconds:
1. Fetch 100 H1 candles from MT5
2. Engineer 16 features (Bollinger, EMA, ATR, RSI, MACD, volume, trend, range position)
3. Run 3 ML models (RF, XGBoost, LightGBM) — binary BUY/SELL prediction
4. 2/3 majority vote required + 60% confidence minimum
5. Filter chain: ATR floor → spread → cooldown → circuit breaker → daily/weekly limits
6. Execute with dynamic SL/TP (1.0× / 1.5× ATR)
7. Monitor: trailing stop, partial profit at 1R, 15-min hold floor
```

## ML Pipeline

### Binary Classification
- **BUY (1):** Long TP hits before long SL within 12 H1 candles (12 hours)
- **SELL (0):** Short TP hits before short SL within 12 H1 candles
- **HOLD:** Dropped from training — these are untradeable sideways candles (~16.5% of data)
- **Spread cost** (0.03%) baked into TP during labeling

### Training
- **365 days** of H1 data (~8,700 candles) — covers multiple BTC regimes
- **Balanced downsampling:** Equal BUY/SELL counts per model (no directional bias)
- **Walk-forward validation:** 5 splits, 6-month train, 1-month test
- **TP=0.5%** ($500 at $100K), **SL=0.4%** ($400). After $30 spread, R:R = 1.17:1

### 16 Features (ranked by importance)
| # | Feature | Description |
|---|---------|-------------|
| 1 | **daily_range_position** | Where price sits in today's high-low range |
| 2 | price_vs_ema200 | Distance from 200-period EMA |
| 3 | price_vs_ema20 | Distance from 20-period EMA |
| 4 | trend_strength | Trend momentum indicator |
| 5 | bb_upper | Bollinger Band upper distance |
| 6 | range_position | Price position in recent range |
| 7 | atr_14 | Average True Range (14) |
| 8 | bb_width | Bollinger Band width (volatility) |
| 9 | rsi_14 | Relative Strength Index |
| 10 | volume_ratio | Volume vs average |
| 11 | macd_signal | MACD signal line |
| 12 | drawdown_pct | Current drawdown % |
| 13 | bb_lower | Bollinger Band lower distance |
| 14 | macd_line | MACD line value |
| 15 | hourly_return | Close-to-close H1 return |
| 16 | price_vs_ema50 | Distance from 50-period EMA |

## Project Structure

```
mt5-trading/
├── btcusd-live/           # LIVE bot directory (currently stopped)
│   ├── trading.py         # Main bot loop (~2200 lines)
│   ├── config.json        # H1, 0.01 lots, 3600s cooldown, 5 max consec losses
│   ├── ml_config.json     # 365d, binary, 16 features, 60% confidence
│   ├── backtest_ml.py     # ML-aware backtester with realistic sizing
│   ├── demo_logger.py     # Structured CSV logging for demo validation
│   ├── ml/                # ML pipeline
│   │   ├── ensemble_predictor.py  # 2/3 majority voting (binary)
│   │   ├── ensemble_trainer.py    # Balanced downsampling + walk-forward
│   │   ├── feature_engineering.py # 16 features + hourly_return + daily_range
│   │   ├── data_preparation.py    # MT5 data fetching (H1/M5 aware)
│   │   └── model_trainer.py       # XGBoost single model trainer
│   ├── models/            # Trained .pkl files + scaler + metadata
│   ├── logs/              # Demo week CSV logs
│   │   ├── trades.csv     # Every closed trade
│   │   ├── signals.csv    # Every ML signal (executed + blocked)
│   │   └── daily_summary.csv  # Daily rollup
│   └── mt5_auth.json      # MT5 credentials (gitignored)
├── btcusd/                # Demo bot (mirror of btcusd-live)
├── xauusd/                # Gold bot (single XGBoost, stopped)
├── eurusd/                # Euro bot (single XGBoost, stopped)
├── vercel-dashboard/      # Cloud dashboard (Vercel + Supabase)
├── auto_retrain.py        # Weekly auto-retrain scheduler
├── ecosystem.config.js    # PM2 process manager config
├── daily_digest.py        # End-of-day performance summary
└── sync_to_supabase.py    # Trade sync to cloud
```

## Configuration

### Runtime (`config.json`)
| Setting | Value |
|---------|-------|
| Timeframe | **H1** |
| Lot Size | 0.01 |
| Trade Cooldown | 3600s (1 hour) |
| Circuit Breaker | 5 consecutive losses = daily shutdown |
| Weekly Limit | 15 trades |
| EMA Trend Filter | Disabled (ML handles trend via features) |
| Momentum Filter | Disabled (ML handles momentum via features) |

### ML (`ml_config.json`)
| Setting | Value |
|---------|-------|
| Training Period | **365 days** |
| Lookahead | 12 candles (12 hours) |
| TP / SL | 0.5% / 0.4% |
| Confidence Threshold | 60% |
| Min Probability Diff | 15% |
| Max Trades/Day | 3 |
| Features | 16 (binary classification) |

## Quick Start

```bash
# Install dependencies
pip install MetaTrader5 scikit-learn xgboost lightgbm pandas numpy joblib requests

# Train models (requires MT5 terminal running)
cd btcusd-live
python train_ml_model.py --refresh --ensemble

# Run backtest
python backtest_ml.py

# Start via PM2
pm2 start ecosystem.config.js --only bot-btcusd-live
pm2 save
```

## Risk Management

| Guard | Detail |
|-------|--------|
| **Circuit Breaker** | 5 consecutive losses → shutdown for rest of MYT day |
| **Daily Limit** | Max 3 trades per day |
| **Weekly Limit** | Max 15 trades per MYT week |
| **ATR Floor** | ATR(14) must be ≥ 50 (skip chop) |
| **Spread Filter** | Max 0.05% of price |
| **1-Hour Cooldown** | Min 3600s between trades |
| **Partial Profit** | Close 50% at 1R, move SL to breakeven |
| **Min Hold** | 15 min before trailing stop can trigger |
| **$0.50 Win Floor** | Micro-wins don't reset circuit breaker counter |

## The Journey (Feb 22 Experiments)

Starting from -5.18% growth, 37.6% WR, 93% SHORT bias:

1. **Removed SELL 2x bias** — class weights equalized
2. **Expanded to 180 days** — more training data
3. **Simplified filters** — disabled redundant momentum/EMA filters
4. **Binary classification** — dropped HOLD, BUY vs SELL only
5. **Trimmed to 15→16 features** — removed 12 low-importance ones
6. **Switched to H1** — massive noise reduction, spread becomes negligible
7. **Balanced downsampling** — forced model to learn patterns, not "BTC goes up"

Result: 36.4% → 51.8% → **63.6% walk-forward accuracy** 🔥

## License

MIT
