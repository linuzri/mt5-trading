# Machine Learning Trading Strategy

## Overview

This ML strategy uses a **Random Forest classifier** to predict BUY/SELL/HOLD signals based on 10 technical indicators calculated from historical BTCUSD data.

## What Machine Learning Does

### Traditional Strategy (Rule-Based)
```python
if RSI < 30:
    signal = "buy"  # Fixed rule
```

### ML Strategy (Pattern Learning)
```python
# Model learns from 90 days of historical data
# Discovers patterns like:
# "When RSI=45, MACD crossing up, ATR high, volume increasing,
#  price usually goes up in next 15 minutes"
signal = model.predict(current_market_features)
```

## How It Works

### Training Phase (One-time setup)
1. **Extract Data**: Downloads 90 days of BTCUSD M1 candles from MT5
2. **Engineer Features**: Calculates 10 technical indicators
3. **Create Labels**:
   - BUY: If price increases >0.5% in next 15 candles
   - SELL: If price decreases >0.5% in next 15 candles
   - HOLD: Otherwise
4. **Train Model**: Random Forest learns patterns
5. **Validate**: Tests on unseen data (70/15/15 split)
6. **Save**: Stores trained model to `models/random_forest_btcusd.pkl`

### Live Trading Phase
1. Bot calculates current market features every loop
2. Feeds features into trained model
3. Model predicts: BUY/SELL/HOLD with confidence score
4. If confidence > 65% and signal != HOLD → Execute trade
5. Otherwise → Skip and wait for next opportunity

## Features Used (10 Indicators)

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `rsi_14` | Relative Strength Index | Overbought/oversold |
| `macd_line` | MACD indicator | Trend momentum |
| `macd_signal` | MACD signal line | Crossover signals |
| `atr_14` | Average True Range | Volatility measure |
| `bb_upper` | Bollinger Band upper | Price envelope |
| `bb_lower` | Bollinger Band lower | Support level |
| `bb_width` | BB width normalized | Volatility squeeze |
| `volume_ratio` | Volume vs average | Buying pressure |
| `price_change_1min` | 1-candle return | Recent momentum |
| `price_change_5min` | 5-candle return | Short-term trend |

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `scikit-learn`: Machine learning library
- `joblib`: Model serialization

### 2. Train the Model
```bash
# First time: Download 90 days of data and train
python train_ml_model.py --refresh

# Re-training (uses cached data)
python train_ml_model.py
```

**Expected output:**
```
📊 Extracting 90 days of BTCUSD data on M1...
✅ Extracted 129,600 candles
🔧 Engineering features...
✅ Added 10 features
🏷️ Creating labels...
   Label distribution:
   - SELL: 25,000 (19.3%)
   - BUY: 24,500 (18.9%)
   - HOLD: 80,100 (61.8%)
🌲 Training Random Forest with 100 trees...
✅ Model training complete

📈 Evaluating on Validation set...
   Accuracy:  0.6234
   Precision: 0.6102
   Recall:    0.6234
   F1-Score:  0.6145

📈 Evaluating on Test set...
   Accuracy:  0.6189
   Precision: 0.6058
   Recall:    0.6189
   F1-Score:  0.6098

🔍 Feature Importance:
   macd_line           : 0.1523
   rsi_14              : 0.1402
   bb_width            : 0.1198
   ...

💾 Saved model to models/random_forest_btcusd.pkl
```

### 3. Configure Trading Bot
Edit `config.json`:
```json
{
  "strategy": "ml_random_forest",
  "symbol": "BTCUSD",
  "timeframe": "M1",
  ...
}
```

### 4. Run Trading Bot
```bash
python trading.py
```

**You'll see ML predictions:**
```
[ML] Model: BUY with 72% confidence | Probabilities: sell:14%, buy:72%, hold:14%
[NOTIFY] BUY order placed, ticket: 123456789, price: 44567.89
```

## Configuration (ml_config.json)

### Key Parameters

```json
{
  "confidence_threshold": 0.65,  // Only trade if >65% confident
  "min_probability_diff": 0.1,   // Avoid near-ties (10% min difference)

  "labeling": {
    "lookahead_candles": 15,     // Predict 15 candles ahead
    "profit_threshold": 0.005    // 0.5% move threshold
  },

  "model_params": {
    "n_estimators": 100,         // 100 decision trees
    "max_depth": 10,             // Max tree depth
    "random_state": 42           // Reproducibility
  }
}
```

### Tuning Tips

**Increase Accuracy** (fewer trades, higher quality):
```json
"confidence_threshold": 0.75,  // More conservative
"min_probability_diff": 0.15
```

**Increase Trade Frequency** (more trades, lower quality):
```json
"confidence_threshold": 0.55,  // More aggressive
"min_probability_diff": 0.05
```

## Performance Metrics

### What Good Looks Like

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Accuracy | <52% | 52-58% | 58-65% | >65% |
| F1-Score | <0.50 | 0.50-0.58 | 0.58-0.65 | >0.65 |
| Precision | <0.55 | 0.55-0.60 | 0.60-0.70 | >0.70 |

**Note:** >50% accuracy is profitable after accounting for fees. Aim for >55% minimum.

### Why 55% Matters
- 55% win rate with 1:1 R:R = Profitable after fees
- 60% win rate = Consistent profitability
- >65% win rate = Excellent (rare in crypto)

## Advantages vs Traditional Strategies

### ✅ Pros
1. **Adapts to Market**: Learns actual patterns from recent data
2. **Multi-Factor**: Considers 10 indicators simultaneously
3. **Probabilistic**: Gives confidence scores (e.g., "72% confident")
4. **Objective**: No emotional bias
5. **Measurable**: Clear accuracy metrics
6. **Improves Over Time**: Retrain weekly with fresh data

### ❌ Cons / Limitations
1. **Black Box**: Hard to explain WHY it made a decision
2. **Overfitting Risk**: Might memorize past instead of learning patterns
3. **Market Regime Changes**: 2025 patterns may not work in 2026
4. **Data Hungry**: Needs 3+ months of quality data
5. **Computational**: Training takes 2-5 minutes
6. **False Confidence**: Can be very wrong with high confidence

## Weekly Retraining (Recommended)

Market conditions change. Retrain weekly to adapt:

```bash
# Every Monday morning
python train_ml_model.py --refresh
```

Add to crontab (Linux/Mac):
```bash
0 8 * * 1 cd /path/to/mt5-trading && python train_ml_model.py --refresh
```

Or Windows Task Scheduler:
- Trigger: Weekly, Monday 8:00 AM
- Action: `python C:\path\to\mt5-trading\train_ml_model.py --refresh`

## Troubleshooting

### Error: "Model file not found"
```bash
# Train the model first
python train_ml_model.py --refresh
```

### Error: "ML modules not available"
```bash
# Install ML dependencies
pip install scikit-learn joblib
```

### Low Accuracy (<55%)
1. **More Data**: Increase `training_period_days` to 180
2. **Better Features**: Add more technical indicators
3. **Tune Model**: Adjust `n_estimators`, `max_depth`
4. **Different Threshold**: Adjust `profit_threshold`

### Too Few Trades
```json
// Lower confidence threshold
"confidence_threshold": 0.60
```

### Too Many Losing Trades
```json
// Higher confidence threshold
"confidence_threshold": 0.70
```

## File Structure

```
mt5-trading/
├── ml/                           # ML module
│   ├── __init__.py
│   ├── data_preparation.py       # Extract MT5 data
│   ├── feature_engineering.py    # Calculate features
│   ├── model_trainer.py          # Train model
│   └── model_predictor.py        # Live predictions
├── models/                        # Trained models (generated)
│   ├── random_forest_btcusd.pkl  # Trained model
│   ├── scaler_btcusd.pkl         # Feature scaler
│   └── random_forest_btcusd_metadata.json  # Model info
├── ml_config.json                # ML configuration
├── train_ml_model.py             # Training script
├── trading.py                    # Main bot (ML integrated)
└── ML_STRATEGY_README.md         # This file
```

## Next Steps: Pattern Recognition (Phase 2)

After ML is working well, add pattern recognition:

### Candlestick Patterns
- Doji, Hammer, Engulfing
- Head & Shoulders, Double Top/Bottom

### Implementation Options
1. **Rule-based**: Detect patterns algorithmically
2. **ML-based**: Use patterns as additional features
3. **Computer Vision**: CNN on candlestick chart images

Would add 15-20 more features to the model.

## FAQ

**Q: Will this make me rich?**
A: No trading strategy guarantees profits. ML improves edge but market risk remains.

**Q: How often should I retrain?**
A: Weekly recommended. Daily if market is very volatile.

**Q: Can I use this on other symbols?**
A: Yes! Edit `symbol` in `ml_config.json`, then retrain.

**Q: Why Random Forest and not Neural Networks?**
A: Random Forest is:
- Easier to train (no hyperparameter hell)
- Less likely to overfit
- Faster predictions
- Interpretable (feature importance)

Neural networks are Phase 3 (after you have more data and experience).

**Q: What if accuracy drops in live trading?**
A: Market regime changed. Retrain immediately with fresh data.

## Support

For issues or questions:
1. Check this README
2. Review `train_ml_model.py` output for errors
3. Check `trade_notifications.log` for ML prediction logs

---

**Remember:** Backtest on demo account for 1-2 weeks before going live!
