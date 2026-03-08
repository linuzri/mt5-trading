# MEMORY.md — AutoTrader Research Program
# BTCUSD Trend-Following Strategy (H4/H1)
# Human edits this file. Agent reads it every loop iteration.
# Last updated: 2026-03-08

---

## MISSION
You are an autonomous optimizer for a BTCUSD trend-following bot running on MT5.
Strategy: H4 EMA alignment for direction → H1 pullback/breakout entry → ATR-based SL/TP.

Each loop: propose ONE param mutation → backtest on last 24h H1 candles → keep or discard.
Do NOT change trading logic. ONLY tune the params listed under MUTABLE PARAMS.

---

## ACTIVE STRATEGY: btcusd (trend_following)

### Current Parameters
```
sl_atr_multiplier : 1.25     # SL = ATR(14) * this
tp_atr_multiplier : 1.75     # TP = ATR(14) * this   (current R:R = 1.40)
atr_period        : 14       # ATR lookback candles (H1)
min_atr           : 300      # Skip trade if H1 ATR < this (BTCUSD points)
h4_ema_fast       : 15       # H4 trend detection EMA fast
h4_ema_slow       : 50       # H4 trend detection EMA slow
h1_ema_period     : 25       # H1 entry signal EMA
lot_size          : 0.01     # Fixed lot size (do NOT increase until win_rate > 58%)
```

### Entry Logic (do not modify — for agent context only)
- H4 bullish (price > EMA20 > EMA50): look for H1 pullback to EMA20 or H1 breakout
- H4 bearish (price < EMA20 < EMA50): look for H1 pullback to EMA20 or H1 breakdown
- H4 neutral: no trade

---

## MUTABLE PARAMS
# Format: param_name: [min, max, step]
# Agent may only change values within these ranges, one step at a time.

sl_atr_multiplier : [1.0, 3.0, 0.25]
tp_atr_multiplier : [1.5, 4.0, 0.25]
h4_ema_fast       : [10, 30, 5]
h4_ema_slow       : [30, 100, 10]
h1_ema_period     : [10, 30, 5]
# min_atr REMOVED — H1 ATR floor is 320 so any value < 320 has no effect; frozen at 300

---

## CONSTRAINTS (hard limits — never violate)
- lot_size is FROZEN at 0.01 — do not mutate until win_rate > 58% for 3 consecutive experiments
- min tp_atr_multiplier: 1.5 (never go below — must maintain positive RR)
- max sl_atr_multiplier: 3.0 (beyond this SL is too wide for account size)
- h4_ema_fast must always be < h4_ema_slow (no crossover inversion)
- one param mutation per experiment only
- backtest window: 168 hours (1 week) of H1 candles

---

## BASELINE METRICS
# Agent compares every new result against these before keeping/discarding.
# Updated only when a KEEP decision is made.

win_rate          : 54.3     # % — 168h backtest 2026-03-08
pnl_per_session   : 82.75    # USD per 168h window (0.01 lot, 35 trades)
max_drawdown      : 0.22     # % of account balance (worst loss streak $110.69 / ~$49,580)
total_trades_avg  : 35       # trades per 168h window
last_updated      : 2026-03-08

---

## DECISION RULES
# Apply strictly — no exceptions.

### KEEP if ALL of:
1. new_win_rate   >= baseline_win_rate - 2.0      # allow up to 2% WR drop if PnL improves
2. new_pnl        >= baseline_pnl * 1.05          # must be >= 5% PnL improvement to keep
3. new_drawdown   <= baseline_drawdown * 1.20     # drawdown can worsen by max 20%
4. new_drawdown   <= 2.0                          # hard drawdown cap (% of account balance)

### DISCARD if ANY of:
1. new_win_rate   < baseline_win_rate - 5.0       # significant win rate drop (>5%)
2. new_pnl        < baseline_pnl                  # any PnL regression at all
3. new_drawdown   > 2.0                           # hard cap: never risk more than 2% of account
4. total_trades   < 5                             # too few signals — not enough data

### EDGE CASE:
- If total_trades == 0: discard immediately, log reason "no signals generated"
- If result is identical to baseline: discard, try a different param next iteration

---

## EXPERIMENT LOG SUMMARY
# Agent updates these counters after every experiment.

total_experiments_run : 206
total_kept            : 6
total_discarded       : 200
keep_rate             : 3%
best_improvement      : "h1_ema_period 20→25: +$26.45 pnl, +2.9% WR"
last_experiment       : "EXP-206 DISCARD: tp_atr_multiplier 1.75→2.25 | WR 54.3%→40.0%, PnL $82.75→$29.16 — wider TP reduced hit rate"
convergence_note      : "SL 1.25 confirmed optimal — both 1.0 and 1.75 discarded. Keep rate at 3% suggests near-convergence."

---

## AGENT INSTRUCTIONS
Read these every loop before proposing anything.

1. Check last_experiment — avoid repeating a param that was just discarded.
2. Look at all MUTABLE PARAMS and pick the one with most room to improve baseline metrics.
3. Prioritize reducing drawdown if max_drawdown > 5.0%.
4. Prioritize increasing win_rate if it's below 52%.
5. Otherwise, target PnL improvement.
6. Propose new_value = old_value ± exactly one step_size.
7. After backtest, apply DECISION RULES strictly.
8. If KEEP: update Current Parameters, BASELINE METRICS, and EXPERIMENT LOG SUMMARY.
9. If DISCARD: only update EXPERIMENT LOG SUMMARY counters.

---

## NOTES FROM HUMAN (Nazri)
# Update this section between sessions with guidance for the next run.

- Demo week 3 (Mar 6-13) — do not touch lot_size
- BTC trend REVERSED Mar 8: EMA50 < EMA200 → DOWNTREND. Bot fires SELLs only.
- Strategy is selective by design — expect 2-8 signals per 168h window
- Baseline confirmed: 54.3% WR, $82.75 PnL, 0.22% DD, 35 trades over 168h (2026-03-08)
- AutoResearch ran 206 experiments, 6 kept (3% keep rate) — near convergence
- SL 1.25 confirmed optimal (both 1.0 and 1.75 discarded)
- All optimised params deployed to live config + bot restarted Mar 8
- min_atr lowered to 250 (from 300) — allows more trades in moderate volatility
- EFFECTIVE params to target (these actually change trade outcomes):
    1. sl_atr_multiplier / tp_atr_multiplier — directly changes win/loss amounts
    2. h4_ema_fast / h4_ema_slow — changes how many H4 bars are trending vs neutral
    3. h1_ema_period — changes where EMA sits, affects pullback/breakout detection
- Primary goal: maintain win_rate above 50% (currently 54.3%)
- Secondary goal: keep drawdown below 0.5% of account balance
- 168h window ensures at least one trending period even in ranging markets