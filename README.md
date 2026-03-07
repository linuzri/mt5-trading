# 🤖 MT5 Trading Bot — BTCUSD H4/H1 Trend Following

Automated Bitcoin trading bot for MetaTrader 5 using **rule-based trend detection** (H4 EMA alignment + H1 pullback/breakout entries) with ATR-based risk management. ML ensemble (RF + XGBoost + LightGBM) trained as quality filter but currently dormant.

**Live Dashboard:** https://trade-bot-hq.vercel.app  
**MQL5 Signal:** https://www.mql5.com/en/signals/2359964

## 🔴 Status (Mar 7, 2026)

| Item | Status |
|------|--------|
| **Live Bot** | ⏸️ STOPPED — pending demo validation |
| **Demo Bot** | ✅ Running from `btcusd/` (trend_following strategy, account 61459537) |
| **Strategy** | H4 trend + H1 pullback/breakout + **H1 momentum filter** (no ML in loop) |
| **ML Model** | Trained as quality filter, loaded but dormant |
| **MQL5 Signal** | ✅ LIVE & APPROVED ($30/month) |
| **Demo Week 3** | March 6-13 — go-live review March 13 |
| **H1 Momentum** | ✅ SELL: lower close+high+below EMA20. BUY: mirror. (Mar 6) |
| **Trailing Stop** | ❌ DISABLED (M1 ATR incompatible with H1 trend holds) |
| **Smart Exit** | ❌ DISABLED (120min max hold kills multi-hour trends) |
| **Position Exit** | SL (1.5× ATR) or TP (2.0× ATR) only |
| **Trade Limits** | Daily 5, Weekly 30 (Mar 7) |
| **ATR Floor** | 300 on H1 (recalibrated from M5 value of 50, Mar 7) |

### Backtest Results (90 days out-of-sample, 0.01 lots, during 26% BTC crash)
| Metric | Value |
|--------|-------|
| Win Rate | **48.5%** |
| Profit Factor | **1.16** |
| Total P/L | **+$78.50** |
| Max Drawdown | **-$118.38** |
| BUY / SELL | 37 (37.8% WR) / 62 (54.8% WR) |
| Trades | 99 over 90 days |

## How It Works

```
On each new H1 candle close:
1. Check H4 EMA20/EMA50 alignment → bullish / bearish / neutral
2. If neutral → skip (no trade in tangled markets)
3. Check H1 for pullback to EMA20 or breakout above/below previous candle
4. H1 momentum confirmation: SELL needs lower close+high+below EMA20; BUY needs higher close+low+above EMA20
5. Filter chain: ATR floor (≥300) → spread → cooldown → circuit breaker → daily(5)/weekly(30) limits
6. Execute with dynamic SL/TP (1.5× / 2.0× ATR, R:R = 1.33:1)
7. Position exits ONLY via SL or TP (no trailing stop, no smart exit, no partial profit)
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
├── btcusd/                # DEMO bot directory (active — demo evaluation week)
│   ├── trading.py         # Main bot loop (~2200 lines)
│   ├── config.json        # H1, 0.01 lots, 3600s cooldown, 5 max consec losses
│   ├── ml_config.json     # 365d, binary, 16 features, 60% confidence
│   ├── state/             # Runtime state
│   │   └── state.json     # Persisted counters (survives PM2 restarts)
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
├── btcusd-live/           # LIVE bot directory (stopped, untouched for isolation)
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
| Daily Limit | 5 trades |
| Weekly Limit | 30 trades (Mar 7, up from 25 for buffer) |
| ATR Floor | 300 on H1 (Mar 7, recalibrated from M5 value of 50) |
| H1 Momentum Filter | Enabled (Mar 6 — SELL: lower close+high+below EMA20) |
| EMA Trend Filter | Removed (deprecated — trend_following uses TrendStrategy H4 EMA20/EMA50) |

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
| **Weekly Limit** | Max 30 trades per MYT week |
| **ATR Floor** | ATR(14) must be ≥ 300 on H1 (skip chop — recalibrated Mar 7 from M5 value of 50) |
| **Spread Filter** | Max 0.05% of price |
| **1-Hour Cooldown** | Min 3600s between trades |
| **Partial Profit** | DISABLED for demo week (was: close 50% at 1R, breakeven SL) |
| **Trailing Stop** | DISABLED (Feb 27) — M1 ATR trailing killed H1 trend trades in minutes |
| **Smart Exit** | DISABLED (Feb 27) — 120min max hold incompatible with 9hr avg holds |
| **State Persistence** | `state.json` — counters survive PM2 restarts |
| **H1 Candle Dedup** | Only evaluates signals on fresh H1 candle close (no redundant polls) |
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
