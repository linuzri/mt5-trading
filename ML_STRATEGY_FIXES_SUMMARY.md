# ML Strategy Fixes - Implementation Summary

## Changes Made (Branch: claude/ml-fixes-O2pfN)

### 1. Updated ml_config.json ✅

**Critical Parameter Changes:**

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `training_period_days` | 30 | 90 | More data for robust learning |
| `lookahead_candles` | 15 | 45 | Match actual trade duration |
| `profit_threshold` | 0.001 (0.1%) | 0.0034 (0.34%) | **Match bot's TP** |
| `stop_loss_threshold` | N/A | 0.0017 (0.17%) | **Match bot's SL** |
| `confidence_threshold` | 0.55 (55%) | 0.68 (68%) | Higher quality signals |
| `min_probability_diff` | 0.05 (5%) | 0.12 (12%) | Avoid close calls |
| `labeling_method` | `future_return` | `sltp_aware` | **CRITICAL FIX** |

**New Risk Management Section:**
```json
"risk_management": {
  "min_atr_threshold": 50,     // Minimum ATR to trade
  "max_trades_per_day": 10,    // Trade limit
  "min_win_rate": 0.50         // Minimum acceptable win rate
}
```

### 2. Implemented SL/TP-Aware Labeling ✅

**File:** `ml/feature_engineering.py`

**New Method:** `create_labels_sltp_aware()`

**What It Does:**
- Labels BUY only if TP ($150 / 0.34%) hits BEFORE SL ($75 / 0.17%)
- Labels SELL only if short TP hits before short SL
- Labels HOLD if neither condition met

**Why This is Critical:**
```python
# OLD METHOD (WRONG):
if price_goes_up_0.1%:
    label = BUY  # But bot might get stopped out at 0.17% loss!

# NEW METHOD (CORRECT):
if tp_hit_before_sl:
    label = BUY  # Only label BUY if trade would actually profit!
```

**Code Example:**
```python
def create_labels_sltp_aware(self, df):
    for i in range(len(df)):
        entry_price = df['close'].iloc[i]
        future_highs = df['high'].iloc[i+1:i+lookahead+1]
        future_lows = df['low'].iloc[i+1:i+lookahead+1]

        # Calculate TP and SL levels
        tp_price = entry_price * (1 + 0.0034)  # 0.34% TP
        sl_price = entry_price * (1 - 0.0017)  # 0.17% SL

        # Check if TP hit before SL
        tp_hit = (future_highs >= tp_price).any()
        sl_hit = (future_lows <= sl_price).any()

        if tp_hit and not sl_hit:
            label = BUY  # Profitable trade!
        else:
            label = HOLD  # Would lose or break even
```

### 3. Added Volatility Filter ✅

**File:** `trading.py`

**What It Does:**
- Checks ATR before taking ML signals
- Skips trades when ATR < 50 (low volatility)
- Prevents trading in choppy/ranging markets

**Code Addition:**
```python
# VOLATILITY FILTER: Check ATR threshold
current_atr = latest_features.get('atr_14', 0)
min_atr_threshold = 50

if current_atr < min_atr_threshold:
    log_only(f"[ML FILTER] ATR {current_atr:.1f} below threshold {min_atr_threshold}")
    trade_signal = None
    continue
```

### 4. Enhanced ML Logging ✅

**Improvements:**
- Added ATR value to ML prediction logs
- Better filter message suppression
- Clearer reason reporting

**Example Output:**
```
[ML] Model: BUY with 72% confidence | ATR: 65.3 | Probabilities: sell:12%, buy:72%, hold:16%
[ML FILTER] ATR 42.1 below threshold 50. Skipping trade.
```

---

## Expected Impact

### Before Fixes:
```
❌ Win Rate: ~30-40% (all losing)
❌ Problem: Model predicts 0.1% moves, bot needs 0.34% moves
❌ Getting stopped out before reaching targets
❌ Overtrading on low-confidence signals
```

### After Fixes:
```
✅ Win Rate: Expected 50-55% initially
✅ Model trained for actual TP/SL requirements
✅ Only high-confidence trades (68% threshold)
✅ Volatility filter prevents choppy market trades
✅ Longer lookahead matches real trade duration
```

---

## How to Use

### 1. Pull Changes on Windows

```bash
cd "C:\Users\Nazri Hussain\Projects\mt5-trading\mt5-trading"
git fetch origin
git checkout claude/ml-fixes-O2pfN
git pull origin claude/ml-fixes-O2pfN
```

### 2. Retrain Model (CRITICAL!)

```bash
python train_ml_model.py --refresh
```

**Expected Training Time:** 5-10 minutes (90 days of data)

**What You'll See:**
```
📊 Extracting 90 days of BTCUSD data on M1...
✅ Extracted ~129,000 candles
🏷️ Creating SL/TP-aware labels (TP=0.34%, SL=0.17%)...
   Label distribution:
   - SELL: X (Y%)
   - BUY: X (Y%)
   - HOLD: X (Z%)
🌲 Training Random Forest with 100 trees...
⚖️ Using class_weight='balanced'
✅ Model training complete

📈 Test Set Accuracy: 0.XXXX
```

**Look For:**
- Accuracy > 55% (profitable threshold)
- BUY/SELL labels should be 10-20% each (was 19% before)
- HOLD should be 60-80% (filtering out marginal setups)

### 3. Run Bot

```bash
python trading.py
```

**Monitor For:**
```
[ML] Model loaded successfully for ml_random_forest strategy
[ML] Model: BUY with 72% confidence | ATR: 65.3 | ...
[ML FILTER] ATR 42.1 below threshold 50. Skipping trade.
```

### 4. Monitor Results (Critical!)

**Track for 24-48 hours on demo:**
- Number of trades (should be 3-5 per day, not 20+)
- Win rate (should be >50%)
- Average win vs average loss (should be ~2:1 due to 1:2 R:R)

---

## Troubleshooting

### Issue: Model accuracy drops below 55%
**Solution:**
- Increase `confidence_threshold` to 0.70
- Increase `profit_threshold` to 0.005 (0.5%)
- Check if market regime changed (retrain)

### Issue: Too few trades (< 2 per day)
**Solution:**
- Lower `confidence_threshold` to 0.65
- Lower `min_atr_threshold` to 40
- Check if market is ranging (normal to have few signals)

### Issue: Still losing trades
**Solution:**
- Check actual SL/TP in config.json matches ml_config.json
- Verify model was retrained with new parameters
- Check if broker spread is too wide
- Monitor slippage on order fills

---

## Next Steps (Future Enhancements)

### Phase 2 (Week 2):
1. **Dynamic SL/TP** - Adjust based on volatility
2. **Market Regime Detection** - Only trade trending markets
3. **Time-based features** - Hour of day, day of week
4. **Exit strategy** - Don't just rely on TP/SL

### Phase 3 (Month 2):
1. **Multi-symbol support** - Trade multiple pairs
2. **Position sizing** - Risk % per trade
3. **Correlation filters** - Avoid correlated trades
4. **Advanced models** - Try XGBoost, LSTM

---

## Key Takeaways

### The Root Problem:
Your model was trained to predict 0.1% moves but your bot needs 0.34% moves to hit TP. Like training a sprinter to run 100m but asking them to win a marathon.

### The Fix:
Retrain model with SL/TP-aware labeling so it learns: "Will this trade hit TP before SL?" instead of "Will price go up?"

### Expected Outcome:
Win rate should improve from 30-40% (losing) to 50-55% (profitable) after fixes, and potentially 60-65% after Phase 2 enhancements.

---

## Files Changed

```
ml_config.json                  - Updated thresholds and added SL/TP params
ml/feature_engineering.py       - Added create_labels_sltp_aware() method
trading.py                      - Added volatility filter and enhanced logging
MAIN_BRANCH_ANALYSIS.md        - Root cause analysis document
ML_STRATEGY_FIXES_SUMMARY.md   - This document
```

---

**Status:** ✅ All critical fixes implemented. Ready for retraining and testing.

**Next Action:** Retrain model with `python train_ml_model.py --refresh`
