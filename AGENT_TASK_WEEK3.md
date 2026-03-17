# Demo Week 3 Fixes — btcusd/ ONLY

CRITICAL SAFETY: ALL changes in btcusd/ folder ONLY. NEVER touch btcusd-live/.

## Fix #0: Log Consolidation
Move all CSV writes to btcusd/logs/ folder. Move state.json to btcusd/state/ folder.
- Create logs/ and state/ dirs on startup with os.makedirs(..., exist_ok=True)
- Define path constants at top: LOGS_DIR, STATE_DIR, TRADE_LOG_PATH, SIGNALS_LOG_PATH, DAILY_SUMMARY_PATH, STATE_PATH
- Update ALL hardcoded file paths in trading.py and any other files that read/write these
- The signals.csv path at ~line 1556 already writes to logs/ - good, keep that
- trade_log.csv currently at btcusd/trade_log.csv - move to btcusd/logs/trade_log.csv
- state.json currently at btcusd/state.json - move to btcusd/state/state.json
- trade_notifications.log can stay at root (it is a runtime log, not a data file)
- Update demo_logger.py if it writes daily_summary.csv

## Fix #1: State Tracking (total_wins/total_losses)
Problem: total_wins and total_losses are always 0 in state.json.
TWO bugs:
1. The counters may not be incremented after trade close - find where trades are closed and add increment logic
2. Line ~1387 in trading.py RESETS total_wins and total_losses to 0 at every day boundary! This wipes daily counters. Change this: keep cumulative total_wins/total_losses across days (never reset them). Only reset daily_trade_count, daily_pl, consecutive_losses, circuit_breaker_triggered at day boundary.

Also fix daily_summary.csv showing trades_taken: 0 every day - likely demo_logger.py is not being called after trade close.

## Fix #2: Signal Dedup — DO NOT IMPLEMENT NEW DEDUP
The existing H1 candle dedup at line ~1533 already works (one evaluation per H1 candle). It is NOT per-direction — it blocks ALL re-evaluation on same candle, which is stricter.
ONLY change needed: persist _last_evaluated_h1_candle to state.json so it survives restarts.
Add it to save_state() and load it on startup.

## Fix #3: H1 Momentum Confirmation for SELL and BUY
Add entry filters. Before entering a trade, confirm H1 momentum matches direction.

For SELL entry, ALL 3 must be true:
- Current H1 close < previous H1 close (lower closes)
- Current H1 high < previous H1 high (lower highs)  
- Current H1 close < H1 EMA20

For BUY entry, ALL 3 must be true:
- Current H1 close > previous H1 close (higher closes)
- Current H1 low > previous H1 low (higher lows)
- Current H1 close > H1 EMA20

Find where trade entry decisions are made (likely in trend_strategy.py or trading.py) and add these checks.
Log blocked entries: warn level, include direction, candle values, and EMA20 value.

## Fix #4: Trade Limits
- Set daily limit to 5 trades/day (add if not exists)
- Change weekly limit from 15 to 25
- Find these in config.json or trading.py constants

## Fix #5: Review Script Circuit Breaker Key
In btcusd/analyze_trades.py or any review script, find cb_keys list and add 'circuit_breaker_triggered' if missing.
If no such file exists, check if there is an mt5_review.py at the project root.

## Verification
After ALL fixes, run: python btcusd/trading.py --help (or similar) to check for syntax errors.
Do NOT actually start the bot.

## What NOT to change
- btcusd-live/ folder — NEVER TOUCH
- Any ML model files
- backtest.py, train_ml_model.py
- The H4 trend detection logic
- ATR-based SL/TP calculations
- Position sizing logic
