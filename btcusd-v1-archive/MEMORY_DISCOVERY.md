# MEMORY_DISCOVERY.md — Strategy Discovery Program
# BTCUSD Strategy Variant Evolution (H4/H1 Trend-Following Base)
# Human edits this file. Agent reads it every loop iteration.
# Last updated: 2026-03-14

---

## MISSION
You are an autonomous strategy inventor for a BTCUSD trend-following bot running on MT5.
Base strategy: H4 EMA alignment for direction → H1 pullback/breakout entry → ATR-based SL/TP.

Each loop: propose ONE new strategy LOGIC variant → generate code → sandboxed backtest → keep or discard.
Do NOT change parameters (autotrader.py handles that). INVENT NEW LOGIC for core functions.

---

## CORE FUNCTIONS (modify these with new logic)

### compute_h4_trend(h4_df, ema_fast, ema_slow)
Current logic: Price > EMA_fast > EMA_slow = bullish, Price < EMA_fast < EMA_slow = bearish
Possible variants:
- Multi-EMA ribbon (3+ EMAs)
- EMA + RSI confirmation
- ATR-adjusted EMA periods
- Session-time weighted EMAs
- Support/resistance levels

### replay_signals(h1_df, h4_trend_series, h1_ema_period, atr_period, min_atr)
Current logic: Pullback to EMA or breakout above/below previous high/low + Bollinger Band pullback
Possible variants:
- RSI oversold/overbought filters
- Volume spike confirmation
- MACD divergence signals
- Multi-timeframe momentum
- Support/resistance bounces

### simulate_trades(signals_df, h1_df, sl_atr_mult, tp_atr_mult, ...)
Current logic: Fixed ATR-based SL/TP, timeout after max_hold_candles
Possible variants:
- Trailing stop modifications
- Partial profit taking
- ATR-based position sizing
- Time-based exit rules
- Smart exit on momentum loss

---

## BASELINE METRICS (current strategy performance)
# Agent compares every variant against these before keeping/discarding.
# Updated only when a KEEP decision is made.

win_rate          : 88.2     # % — 168h backtest (long-only mode, current market)
pnl               : 145.69   # USD per 168h window (0.01 lot, 17 trades)
max_drawdown      : 0.01     # % of account balance
total_trades      : 17       # trades per 168h window
avg_hold_candles  : 9.6      # average position duration
last_updated      : 2026-03-14

---

## DECISION RULES (apply strictly)

### KEEP if ALL of:
1. new_win_rate   >= baseline_win_rate - 3.0      # allow up to 3% WR drop for logic experiments
2. new_pnl        > baseline_pnl + 8.0            # must improve PnL by at least $8 (higher threshold for logic changes)
3. new_drawdown   <= baseline_drawdown * 1.30     # drawdown can worsen by max 30% (more lenient for variants)
4. new_drawdown   <= 2.5                          # hard drawdown cap (% of account balance)
5. total_trades   >= 5                            # minimum signal count (higher than param optimization)

### DISCARD if ANY of:
1. new_win_rate   < baseline_win_rate - 8.0       # significant win rate drop (>8%)
2. new_pnl        <= baseline_pnl                 # any PnL regression at all
3. new_drawdown   > 2.5                           # hard cap: never risk more than 2.5% of account
4. total_trades   < 3                             # too few signals generated
5. execution_error                                # generated code failed to run

### EDGE CASES:
- If total_trades == 0: discard immediately, reason "no signals generated"
- If variant generates >50 trades: discard, reason "signal spam - too aggressive"
- If generated code has syntax errors: discard, reason "invalid code generation"
- If backtest takes >300s: discard, reason "inefficient logic - timeout"

---

## CURRENT MODE CONSTRAINTS
- **LONG-ONLY:** Bot only takes BUY signals (SELL signals blocked by config)
- **D1 TREND FILTER:** BUY blocked when D1 EMA50 < EMA200 (daily downtrend)
- All variants must respect long-only mode - no point optimizing SELL logic
- Expect ~5-15 trades per 168h window (fewer than bidirectional strategy)
- BTC currently in UPTREND: EMA50 > EMA200 on daily chart

---

## SUCCESSFUL VARIANTS (kept variants)
# Updated after each KEEP decision

### VARIANT 1: bollinger_pullback
**Date:** 2026-03-14
**Function Modified:** replay_signals  
**Description:** BUY when price pulls back to lower Bollinger Band (oversold) but stays above EMA20 in bullish H4 trend  
**Performance:** 88.2% WR, $145.69 PnL, 0.01% DD (17 trades)  
**Key Logic:** Added third entry condition - Bollinger Band lower band pullback alongside existing EMA pullback and breakout signals  
**Rationale:** Bollinger Bands identify oversold conditions more dynamically than fixed EMA levels for high-probability BUY setups  

---

## EXPERIMENT COUNTERS
# Agent updates these after every experiment

total_experiments_run : 1
total_kept            : 1
total_discarded       : 0
keep_rate             : 100.0%
best_improvement      : "+65.29 USD PnL, +17.6% WR (bollinger_pullback)"
last_experiment       : "bollinger_pullback - KEPT"

---

## VARIANT CATEGORIES TO EXPLORE
# Agent should cycle through these areas to ensure diverse exploration

### 1. CONFIRMATION FILTERS (high priority)
- RSI oversold confirmation for BUY signals (RSI < 30)
- Volume spike detection (volume > 2x average)
- MACD bullish divergence detection
- Stochastic oversold confirmation
- ~~Bollinger band position filters~~ **DONE: bollinger_pullback**

### 2. MULTI-TIMEFRAME LOGIC (medium priority)
- M15 momentum confirmation for H1 entries
- M5 micro-trend alignment
- D1 trend strength measurement
- Weekly pivot point awareness
- Multiple timeframe EMA ribbon

### 3. DYNAMIC EXIT STRATEGIES (medium priority)
- Chandelier exit system
- ATR-based trailing stops
- Profit target scaling based on trend strength
- Time-decay exit rules
- Momentum-based early exit

### 4. MARKET CONDITION ADAPTORS (low priority)
- Session-time filters (London/NY overlap)
- Volatility regime detection
- News event avoidance
- Holiday trading restrictions
- Market hour risk adjustments

### 5. ADVANCED ENTRIES (low priority)
- Support/resistance level detection
- Fibonacci retracement entries
- Flag/pennant pattern recognition
- Gap fill strategies
- Order flow imbalance detection

---

## AGENT INSTRUCTIONS
Read these before every variant generation:

1. **Diversity First:** Avoid generating similar variants. Cycle through categories above.
2. **Code Quality:** Generated function must be complete, syntactically correct, and executable.
3. **Long-Only Aware:** All logic should optimize for BUY signals only.
4. **Keep It Simple:** Start with simple modifications. Complex variants can evolve later.
5. **Maintain Compatibility:** New function must accept same parameters as original.
6. **Document Logic:** Include brief comments explaining new logic in generated code.
7. **Test Edge Cases:** Consider what happens with insufficient data, missing indicators.
8. **Performance Aware:** Avoid computationally expensive operations that slow backtests.

## VARIANT GENERATION PROCESS:
1. Read recent experiment history to avoid repeating variants
2. Pick unexplored category from above list
3. Design specific modification to one core function
4. Generate complete, working Python function code
5. Include rationale explaining expected improvement
6. Ensure code handles edge cases gracefully

---

## NOTES FROM HUMAN (Nazri)
# Update this section with guidance for the next discovery session

- **2026-03-14: INITIAL SETUP** — Strategy discovery system deployed.
  - Base strategy is current autotrader-optimized trend_following with 70.6% WR
  - System should invent LOGIC improvements, not parameter changes
  - Focus on long-only optimizations (BUY signal improvements only)
  - ~~First successful variant will establish evolutionary baseline~~ **DONE: bollinger_pullback**
- **BREAKTHROUGH:** First variant (bollinger_pullback) achieved 88.2% WR (+17.6%) and $145.69 PnL (+$65)
- **New Baseline:** System now has high-performance baseline to beat (88.2% WR, $145.69 PnL)
- **Target Areas:** RSI confirmation, volume filters, better exit rules, multi-timeframe logic
- **Avoid:** Complex pattern recognition (too error-prone), SELL logic (unused), repeating Bollinger logic
- **Success Criteria:** Beat 88.2% WR or $145.69 PnL while keeping DD < 2%
- **Integration:** bollinger_pullback can be manually reviewed and integrated into main strategy
- **Timeline:** Continue discovery for ~19 more experiments to establish variant landscape