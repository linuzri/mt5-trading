# MT5-Trading Python Bot

A robust, fully automated MetaTrader 5 (MT5) trading bot in Python supporting multiple strategies including **Machine Learning**, advanced risk management, and comprehensive notifications/logging. Supports both **Forex and Crypto CFDs** (BTCUSD, ETHUSD, etc.).

## Features

### Core Trading
- **Automated trading on MT5** (Pepperstone, IC Markets, or any MT5 broker)
- **Configurable symbol** - Trade Forex (EURUSD) or Crypto (BTCUSD, ETHUSD)
- **Multiple strategies:** MA Crossover, RSI, MACD, Bollinger Bands, **ML Random Forest**
- **Scalping support:** M1 timeframe for quick trades
- **All settings configurable** via `config.json`

### Machine Learning Strategy
- **Random Forest classifier** trained on 30 days of historical data
- **24 technical indicators** as features (RSI, MACD, ATR, Bollinger Bands, Williams %R, EMA trend, etc.)
- **Confidence-based trading** - Only trades when model confidence exceeds threshold
- **Daily auto-retraining** - Model retrains every day at 8:00 AM ET with fresh data
- **Class-balanced training** - Handles imbalanced BUY/SELL/HOLD labels
- **Session-aware features (NEW)** - Hour encoding helps model learn time-of-day patterns

### Smart Optimization
- **Daily auto-training** - ML model retrains every day at 8:00 AM US Eastern
- **Multi-timeframe confirmation** - Only trades when lower and higher timeframes agree
- **ATR volatility filter** - Trades only when volatility is above threshold

### Risk Management
- **Configurable SL/TP** - Stop loss and take profit in price units
- **Trailing stop loss** - ATR-based trailing stop management
- **Daily loss/profit limits** - Auto-pause when limits reached
- **News event avoidance** - Placeholder for news API integration

### Defensive Trading (NEW)
- **Spread filter** - Skips trades when spread exceeds threshold (protects against low liquidity)
- **Loss cooldown** - Waits X minutes after a losing trade before next entry (prevents revenge trading)
- **Consecutive loss circuit breaker** - Pauses trading after X consecutive losses (protects against choppy markets)

### Notifications
- **Telegram alerts** - Only for trades (BUY/SELL), not spam
- **Trade statistics** - Win/loss count, win rate, and session P/L after each trade
- **Close reason tracking** - Shows if trade closed by SL Hit, TP Hit, or Reversal
- **Hourly heartbeat** - Confirms bot is running with balance info
- **Trade summaries** - Daily and weekly P/L summaries
- **Critical alerts** - Max loss/profit triggers immediate notification

### Logging
- **Console output** - Real-time status
- **File logging** - All events to `trade_notifications.log`
- **Trade history** - Closed trades to `trade_log.csv`

## Requirements

- Python 3.8+
- MetaTrader 5 Terminal (with AutoTrading enabled)
- MT5 Broker account (demo or live)

```sh
pip install -r requirements.txt
```

## Quick Start

### 1. Clone and Install
```sh
git clone https://github.com/linuzri/mt5-trading.git
cd mt5-trading
pip install -r requirements.txt
```

### 2. Create `mt5_auth.json`
```json
{
  "login": 123456789,
  "password": "YOUR_MT5_PASSWORD",
  "server": "Pepperstone-Demo",
  "telegram": {
    "bot_name": "@your_telegram_bot",
    "api_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "chat_id": 123456789
  }
}
```

### 3. Train ML Model (for ML strategy)
```sh
python train_ml_model.py --refresh
```

### 4. Configure `config.json`
```json
{
  "symbol": "BTCUSD",
  "sl_pips": 75,
  "tp_pips": 150,
  "timeframe": "M1",
  "higher_timeframe": "M15",
  "strategy": "ml_random_forest",
  "enable_trailing_stop": false,
  "max_daily_loss": 100,
  "max_daily_profit": 200
}
```

### 5. Run the Bot
```sh
python trading.py
```

## Configuration Options

| Setting | Description | Example |
|---------|-------------|---------|
| `symbol` | Trading instrument | `"BTCUSD"`, `"EURUSD"` |
| `timeframe` | Primary chart timeframe | `"M1"`, `"M5"`, `"H1"` |
| `higher_timeframe` | Trend confirmation timeframe | `"M15"`, `"H1"` |
| `strategy` | Trading strategy | `"ml_random_forest"`, `"ma_crossover"`, `"rsi"`, `"macd"`, `"bollinger"` |
| `sl_pips` | Stop loss in price units | `75` ($75 for crypto) |
| `tp_pips` | Take profit in price units | `150` ($150 for crypto) |
| `enable_trailing_stop` | Enable ATR trailing stop | `true`, `false` |
| `atr_period` | ATR calculation period | `14` |
| `atr_multiplier` | Trailing stop ATR multiplier | `1.5` |
| `min_atr` | Minimum ATR to trade (0=disabled) | `0` |
| `max_daily_loss` | Max daily loss before pause (USD) | `100` |
| `max_daily_profit` | Max daily profit before pause (USD) | `200` |
| `max_spread_percent` | Max spread as % of price (0=disabled) | `0.05` |
| `loss_cooldown_minutes` | Minutes to wait after a loss (0=disabled) | `15` |
| `max_consecutive_losses` | Pause after X consecutive losses (0=disabled) | `3` |

## Machine Learning Strategy

### How It Works

**Traditional Strategy (Rule-Based):**
```python
if RSI < 30:
    signal = "buy"  # Fixed rule
```

**ML Strategy (Pattern Learning):**
```python
# Model learns from 30 days of historical data
# Discovers complex patterns across 10 indicators
signal = model.predict(current_market_features)
```

### Features Used (10 Indicators)

| Feature | Description |
|---------|-------------|
| `rsi_14` | Relative Strength Index (overbought/oversold) |
| `macd_line` | MACD indicator (trend momentum) |
| `macd_signal` | MACD signal line (crossover signals) |
| `atr_14` | Average True Range (volatility) |
| `bb_upper` | Bollinger Band upper (price envelope) |
| `bb_lower` | Bollinger Band lower (support level) |
| `bb_width` | BB width normalized (volatility squeeze) |
| `volume_ratio` | Volume vs average (buying pressure) |
| `price_change_1min` | 1-candle return (recent momentum) |
| `price_change_5min` | 5-candle return (short-term trend) |

### ML Configuration (ml_config.json)

```json
{
  "model_type": "random_forest",
  "data_collection": {
    "training_period_days": 30
  },
  "prediction": {
    "confidence_threshold": 0.55,
    "min_probability_diff": 0.5
  },
  "labeling": {
    "lookahead_candles": 15,
    "profit_threshold": 0.001
  }
}
```

**Tuning Tips:**
- Increase `confidence_threshold` (0.60-0.70) for fewer but higher quality trades
- Decrease `confidence_threshold` (0.50-0.55) for more trades

### Training the Model

```sh
# First time or refresh with latest data
python train_ml_model.py --refresh

# Re-training (uses cached data)
python train_ml_model.py
```

**Expected Output:**
```
Label distribution:
- SELL: 8600 (20.0%)
- BUY: 8991 (20.9%)
- HOLD: 25514 (59.2%)

Cross-Validation Accuracy: 0.5803
Test Set Accuracy: 0.5934

Classification Report:
        SELL: precision=0.40, recall=0.59
        BUY:  precision=0.45, recall=0.52
        HOLD: precision=0.80, recall=0.62
```

### Performance Metrics

| Metric | Poor | Acceptable | Good |
|--------|------|------------|------|
| Accuracy | <52% | 52-58% | >58% |
| BUY/SELL Recall | <30% | 30-50% | >50% |

**Note:** >55% accuracy with balanced recall is profitable after fees.

## How It Works

### Trading Flow
```
Every 60 seconds:
  1. Check market status
  2. Send hourly heartbeat (once per hour)
  3. Fetch price data (M1 bars)
  4. Check higher timeframe trend (M15)
  5. Apply ATR volatility filter
  6. Generate signal from strategy (ML or traditional)
  7. Execute trade if signal found
  8. Manage trailing stops
  9. Log to file (Telegram only for trades)
```

### Daily ML Training (8:00 AM ET)
```
  1. Run train_ml_model.py --refresh
  2. Download latest 30 days of data
  3. Engineer 10 technical features
  4. Train Random Forest with class balancing
  5. Save model to models/random_forest_btcusd.pkl
  6. Reload model in trading bot
  7. Send Telegram notification
```

## Telegram Notifications

**What sends to Telegram:**
- BUY/SELL orders placed
- Trade results with win/loss statistics
- Close reason (SL Hit, TP Hit, Reversal)
- Trailing stop updates
- Hourly heartbeat with balance
- Daily ML training results
- Max loss/profit alerts

**What does NOT send to Telegram:**
- "No trade signal" messages
- Filter messages (ATR, trend)
- Error messages
- Market closed messages

### Example Notifications
```
[HEARTBEAT] Bot running. Balance: $50000.10 | Strategy: ml_random_forest | BTCUSD

[ML] Model: BUY with 62% confidence | Probabilities: sell:18%, buy:62%, hold:20%
[NOTIFY] BUY order placed, ticket: 224067017, price: 94567.89
[BALANCE] Account balance after BUY: $50000.10

[TRADE WIN] BUY (TP Hit) | Entry: 94567.89 | Exit: 94717.89 | P/L: $150.00
[STATS] Wins: 5 | Losses: 2 | Win Rate: 71.4% | Session P/L: $425.00

[TRADE LOSS] SELL (SL Hit) | Entry: 94500.00 | Exit: 94575.00 | P/L: -$75.00
[STATS] Wins: 5 | Losses: 3 | Win Rate: 62.5% | Session P/L: $350.00

[AUTOMATION] ML model retrained successfully
```

## Files

| File | Description |
|------|-------------|
| `trading.py` | Main trading bot with all logic |
| `train_ml_model.py` | ML model training script |
| `backtest.py` | Traditional strategy optimization |
| `config.json` | Trading parameters |
| `ml_config.json` | ML model configuration |
| `mt5_auth.json` | Credentials (DO NOT COMMIT) |
| `trade_notifications.log` | All events log |
| `trade_log.csv` | Closed trades history |

### ML Module Files

| File | Description |
|------|-------------|
| `ml/data_preparation.py` | Extract MT5 data |
| `ml/feature_engineering.py` | Calculate technical features |
| `ml/model_trainer.py` | Train Random Forest model |
| `ml/model_predictor.py` | Live predictions |
| `models/random_forest_btcusd.pkl` | Trained model (generated) |
| `models/scaler_btcusd.pkl` | Feature scaler (generated) |

## Supported Brokers

Tested with:
- **Pepperstone** (recommended for crypto CFDs)
- **IC Markets**
- **MetaQuotes Demo**

The bot auto-detects the correct order filling mode (FOK/IOC) for each broker.

## Crypto vs Forex

| Setting | Forex (EURUSD) | Crypto (BTCUSD) |
|---------|----------------|-----------------|
| `symbol` | `"EURUSD"` | `"BTCUSD"` |
| `sl_pips` | `0.003` (30 pips) | `75` ($75) |
| `tp_pips` | `0.006` (60 pips) | `150` ($150) |
| `timeframe` | `"M5"` | `"M1"` (scalping) |
| Market hours | Mon-Fri | 24/7 |

## Security

- Never commit `mt5_auth.json`
- Always test with demo account first
- Enable AutoTrading in MT5 terminal
- Monitor Telegram for critical alerts

## Troubleshooting

### "AutoTrading disabled"
- Click AutoTrading button in MT5 toolbar (should be green)
- Check Tools > Options > Expert Advisors

### "Unsupported filling mode"
- Bot auto-detects correct mode
- If issues persist, check broker symbol settings

### "Model file not found"
```sh
python train_ml_model.py --refresh
```

### "ML modules not available"
```sh
pip install scikit-learn joblib
```

### Low ML Accuracy (<55%)
1. Adjust `profit_threshold` in ml_config.json
2. Increase `training_period_days` (if broker supports)
3. Tune `confidence_threshold`

### No historical data
- M1 data limited to ~30 days on most brokers
- Use M5 timeframe for longer history

---

**Ready to trade!** Run `python trading.py` and let the bot find profitable opportunities.

For detailed ML documentation, see `ML_STRATEGY_README.md`.
