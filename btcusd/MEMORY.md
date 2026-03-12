# MEMORY.md — AutoTrader Research Program
# BTCUSD Trend-Following Strategy (H4/H1)
# Human edits this file. Agent reads it every loop iteration.
# Last updated: 2026-03-12

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
sl_atr_multiplier : 1.5      # SL = ATR(14) * this
tp_atr_multiplier : 2.75     # TP = ATR(14) * this   (current R:R = 1.83)
atr_period        : 14       # ATR lookback candles (H1)
min_atr           : 300      # Skip trade if H1 ATR < this (BTCUSD points)
h4_ema_fast       : 15       # H4 trend detection EMA fast
h4_ema_slow       : 80       # H4 trend detection EMA slow
h1_ema_period     : 25       # H1 entry signal EMA
lot_size          : 0.01     # Fixed lot size (do NOT increase until win_rate > 58%)
```

### Entry Logic (do not modify — for agent context only)
- H4 bullish (price > EMA20 > EMA50): look for H1 pullback to EMA20 or H1 breakout
- H4 bearish: BLOCKED (long_only_mode = true, see MODE section below)
- H4 neutral: no trade

### MODE (deployed 2026-03-12, PR #48)
- **long_only_mode: TRUE** — ALL SELL signals are blocked. Bot only takes BUY trades.
- **d1_trend_filter: TRUE** — D1 EMA50/EMA200 as master trend filter.
  - BUY blocked when D1 EMA50 < EMA200 (daily downtrend)
  - SELL blocked when D1 EMA50 > EMA200 (daily uptrend) — redundant with long_only but defence-in-depth
- **Why:** Demo showed 24% WR with 7/8 SELL trades losing money (shorting into bounces during uptrend). BUY trades were profitable. Long-only eliminates the losing side.
- **Impact on research:** Expect fewer trades per window (~5-8 instead of 13). All signals are BUY-only. Optimize for long-side performance only.

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

win_rate          : 69.2     # % — 168h backtest 2026-03-08
pnl_per_session   : 140.28   # USD per 168h window (0.01 lot, 13 trades)
max_drawdown      : 0.06     # % of account balance (worst loss streak $31.61 / ~$49,580)
total_trades_avg  : 13       # trades per 168h window
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
4. total_trades   < 3                             # too few signals — lowered from 5 (long-only generates fewer trades)

### EDGE CASE:
- If total_trades == 0: discard immediately, log reason "no signals generated"
- If result is identical to baseline: discard, try a different param next iteration

---

## EXPERIMENT LOG SUMMARY
# Agent updates these counters after every experiment.

total_experiments_run : 212
total_kept            : 12
total_discarded       : 200
keep_rate             : 5.7%
best_improvement      : "tp_atr_multiplier 2.5→2.75: +$15.93 pnl, same WR, same trades"
last_experiment       : "EXP-212 KEEP: tp_atr_multiplier 2.5→2.75 | WR 69.2%→69.2%, PnL $124.35→$140.28 — wider TP increased profits with identical win rate"
convergence_note      : "TP expansion continues to yield gains. Risk/reward now at optimal 1.83 ratio."
mode_change_note      : "2026-03-12: LONG-ONLY MODE + D1 TREND FILTER deployed. Previous 212 experiments were bidirectional. Baseline needs re-validation under new long-only regime. First post-change run will establish new baseline."

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

- Demo week 3 extended — do not touch lot_size. Review March 22.
- **2026-03-12: LONG-ONLY MODE DEPLOYED.** Bot ONLY takes BUY signals now.
  - Reason: Demo showed 24% WR overall, but SELL trades were the problem (7/8 lost money shorting into uptrend bounces). BUY trades were solid winners.
  - D1 trend filter also active: blocks BUY when daily EMA50 < EMA200.
  - BTC currently in UPTREND (EMA50: 69,836 > EMA200: 69,005 as of Mar 12).
- **IMPORTANT FOR RESEARCH:** All experiments now run in long-only context.
  - SELL signals will never fire — don't waste cycles on params that only affect shorts.
  - Expect ~5-8 trades per 168h window instead of 13 (half the signals removed).
  - Minimum trade threshold lowered to 3 (from 5) to account for fewer signals.
  - First run should establish a new LONG-ONLY BASELINE before optimizing.
- Previous 212 experiments were bidirectional — those results are still valid context but the optimization surface has changed.
- AutoResearch ran 212 experiments, 12 kept (5.7% keep rate) — was near convergence on bidirectional strategy
- TP 2.75 confirmed optimal for bidirectional — may shift under long-only (longs ride trends longer, consider testing 3.0)
- min_atr at 250 (lowered from 300 on Mar 8) — keep for now
- EFFECTIVE params to target (long-only context):
    1. tp_atr_multiplier — BUY trades can run further in uptrends, test wider TPs
    2. sl_atr_multiplier — tighter SL might work for longs (trend support = natural floor)
    3. h4_ema_fast / h4_ema_slow — tune for bullish-only alignment detection
    4. h1_ema_period — optimize pullback entry for long entries specifically
- Primary goal: establish long-only baseline, then optimize WR and PnL
- Secondary goal: keep drawdown below 0.5% of account balance
- 168h window ensures at least one trending period even in ranging markets