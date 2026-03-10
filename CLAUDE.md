# CLAUDE.md - AI Agent Context

This file provides context for AI agents working on this codebase.

## Project Overview

Automated MT5 trading bot. **Path B architecture**: rule-based trend-following (H4 direction + H1 pullback/breakout entries) with ATR-based risk management. ML ensemble trained but dormant (quality filter, not in trading loop).

### Current Status (Mar 10, 2026)
- **LIVE:** Account 51439249 — **STOPPED** (pending demo validation)
- **DEMO:** Account 61459537 — Running from `btcusd/` (PM2: `bot-btcusd`)
- **Strategy:** `trend_following` — H4 EMA15/80 alignment + H1 pullback/breakout entries + **H1 momentum confirmation** (Mar 6)
- **Demo Week 3:** March 6-13 (11 fixes deployed, review March 13)
- **AutoResearch:** Karpathy-style autonomous param optimizer — 260+ experiments, 11 kept, deployed Mar 10. Fully autonomous deploy pipeline. Weekly cron via PM2 (Sunday 11PM MYT).
- **ML Model:** Trained (trade quality filter), loaded but NOT in trading loop
- **MQL5 Signal:** https://www.mql5.com/en/signals/2359964 — LIVE, APPROVED ✅
- **Auto-merge PRs:** Granted — merge directly without review
- **BTC Trend:** DOWNTREND (EMA50 < EMA200 as of Mar 8) — bot fires SELLs only

### Demo Week 2 Results (March 2-6) — FAILED
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Win Rate | 26.7% | >45% | ❌ |
| Profit Factor | 0.34 | >1.1 | ❌ |
| Net PnL | -$51.48 | Positive | ❌ |
| SELL Net PnL | -$47.95 | Positive | ❌ Critical |

Root causes: signal spam (127/day), no H1 momentum filter (SELL into rising H1), state tracking blind (total_wins/losses always 0).

### Mar 6 — Demo Week 3 Fixes (PR #42)
1. **Log consolidation** — All CSVs → `btcusd/logs/`, state → `btcusd/state/`
2. **State tracking** — `total_wins`/`total_losses` cumulative (was resetting daily = circuit breaker blind)
3. **H1 dedup persistence** — `_last_evaluated_h1_candle` survives PM2 restarts via state.json
4. **H1 momentum filter** — SELL requires lower close + lower high + below EMA20. BUY mirror.
5. **Trade limits** — Daily 5, weekly 25 (was 15)
6. **Review script** — No cb_keys list found (no change needed)

### Mar 7 — Demo Week 3 Tuning (PR #43)
1. **`min_atr` recalibrated** — 50 → 300. Old value was M5-era; H1 ATR(14) ranges 400–1500. Now actively filters chop.
2. **H1 dedup state save on every poll** — `save_state()` called immediately after `_last_evaluated_h1_candle` update (was only on trade close). Prevents same-candle re-evaluation after mid-candle PM2 restart.
3. **`_trend_strategy` singleton cleared on hot-reload** — `config.pop('_trend_strategy', None)` in `reload_config_and_strategy()`. Ensures fresh `TrendStrategy` instance after config reload.
4. **Dead EMA filter keys removed** — `enable_ema_trend_filter`, `ema_fast_period`, `ema_slow_period` and comments removed from `config.json`. `trend_following` never reads them.
5. **Weekly trade limit** — 25 → 30. Provides buffer when daily limit hits early in the week.

### Feb 27 — Premature Exit Bug Fix (Critical)

**What happened:** First demo trade exited in 25 minutes with -$0.18 P/L (spread cost only). Backtest average hold was 9.2 hours. Root cause: M1 ATR trailing stop was ratcheting SL to near-entry after 15min min_hold_minutes, and smart_exit max_hold=120min would kill any trade after 2 hours.

**Fixes applied (both `btcusd/` and `btcusd-live/`):**
1. **Trailing stop DISABLED** — `enable_trailing_stop: false` in config.json. M1 ATR trailing is incompatible with H1 trend holds.
2. **Smart exit DISABLED** — `smart_exit.enabled: false`. Max hold 120min kills trades that should hold 9+ hours.
3. **H1 candle dedup** — Added to `btcusd/trading.py`. Only evaluates signals on fresh H1 candle close. Eliminates log spam and prevents edge-case duplicate entries within same candle.

**Position management now:** Trades exit ONLY via their original SL (1.5× ATR) or TP (2.0× ATR). No trailing, no smart exit, no partial profit.

### Feb 26 — Path B Pivot (Major Architecture Change)

**What happened:** Found critical look-ahead bias in `daily_range_position` feature. The 63.6% walk-forward accuracy was fake — model saw future data. After fixing, honest accuracy was ~52%. Ran 4 experiments trying to make ML predict BUY/SELL — all failed to achieve balanced recall.

**The pivot:** Stopped asking ML "should I BUY or SELL?" and switched to simple rule-based trend-following. ML retrained as a quality filter ("is this a good trend trade?") but even that only added marginal value in backtest, so deployed without it.

**Journey summary:**
- Started: ML ensemble scalping on M5, 93% short bias, -5.18% growth, 101 trades/week
- Now: Simple trend-following on H4/H1, no ML in loop, 3 trades/day max
- Discovered: SELL 2x class weight → 93% short bias, 30 days training too little, look-ahead bias inflated accuracy 50%→64%, ML couldn't predict direction with balanced recall

**Backtest (90 days OOS, 0.01 lots, during 26% BTC crash):**
- 99 trades, 48.5% WR, PF 1.16, +$78.50 P/L
- Survived hostile market conditions (BTC $92K → $68K)

### Go-Live Criteria (Demo Week 3, review March 13)
- Win rate ≥ 45%
- Profit factor ≥ 1.1
- SELL win rate ≥ 35%
- Signals per day ≤ 8 (after dedup)
- Net PnL positive
- `state.json` total_wins matches actual winners (data integrity)
- Trades per day ≤ 5 (daily limit)
- **Before live: change max_consecutive_losses from 5 → 3**
- **Nazri must give explicit approval**

### Key Lessons Learned
- **NEVER use `groupby` on calendar dates in features** — leaks future data within the same day
- **ML BUY/SELL prediction is extremely hard** on BTC — 4 experiments, best was 53.76% with terrible recall balance
- **Simple trend-following beats complex ML** when the ML can't achieve >55% accuracy
- **Always verify backtest P/L math** — had 100x error in lot size calculation
- **Backtest with spread costs** — $30 spread on BTCUSD eats into profits significantly

## Architecture

```
btcusd-live/trading.py ─→ MetaTrader 5 API ─→ Pepperstone (demo: 61459537)
         │
         ├─→ logs/signals.csv (every ML signal, executed + blocked)
         ├─→ logs/trades.csv (every closed trade with full details)
         ├─→ logs/daily_summary.csv (daily rollup metrics)
         ├─→ blocked_signals.csv (legacy blocked signal log)
         ├─→ trade_notifications.log (verbose activity log)
         ├─→ demo_logger.py (structured CSV logging module)
         ├─→ Telegram notifications
         └─→ sync_to_supabase.py ─→ Supabase (cloud DB)
                                          │
              vercel-dashboard/ ─→ Cloud UI (Vercel)
```

## Bot Loop (H1 Binary Classification)

1. Every 60 seconds: fetch 100 H1 candles from MT5
2. Calculate **16 features** (bb_upper, bb_lower, range_position, price_vs_ema200, atr_14, drawdown_pct, bb_width, trend_strength, macd_signal, volume_ratio, price_vs_ema50, macd_line, rsi_14, price_vs_ema20, hourly_return, daily_range_position)
3. Run ensemble prediction: RF, XGB, LGB each vote **BUY or SELL** (binary, no HOLD class)
4. **2/3 majority agreement** — if models disagree, no trade (HOLD = disagreement)
5. Apply filter chain (in order):
   a. Confidence gate (≥60%, session-adjusted: +10% off-hours)
   b. Min probability diff (≥15% between BUY and SELL)
   c. ATR floor (≥50)
   d. Spread filter (≤0.05% of price)
   e. Trade cooldown (3600s = 1 hour between trades)
   f. Circuit breaker (5 consecutive losses = daily shutdown)
   g. Daily limit (3 trades/day max)
   h. Weekly limit (15 trades/week)
6. If signal passes ALL filters → execute with dynamic SL/TP (1.0×/1.5× ATR)
7. Monitor open positions: trailing stop, partial profit at 1R (after 15min hold floor)
8. Log to `logs/signals.csv` (all signals) and `logs/trades.csv` (closed trades)

## ML Pipeline Details

### Binary Classification
- **Classes:** SELL (0), BUY (1) — no HOLD class
- **Label method:** `sltp_aware` — BUY if long TP hits before SL, SELL if short TP hits before SL
- **HOLD rows dropped** from training (~16.5% of data = untradeable sideways candles)
- **Spread cost** (0.03%) baked into TP calculation during labeling

### Training Data Balancing
- Each model (RF, XGB, LGB) trains on **downsampled** data with equal BUY/SELL counts
- `_balance_training_data()` randomly samples minority class count from majority
- No class_weight parameters — let balanced data speak

### Walk-Forward Validation
- 5 splits, 6-month train window, 1-month test window
- Auto-detects candles/day from data density
- Reports mean accuracy ± std, min/max, break-even threshold (0.48)

### Feature Importance Analysis
- Prints top 15 + bottom features after training
- `daily_range_position` consistently #1 (where price sits in today's high-low range)

## Key Files

### Live Bot (currently stopped)
- `btcusd-live/trading.py` — Main bot loop (~2200 lines). ALL trading logic.
- `btcusd-live/config.json` — Runtime config (H1, 0.01 lots, 3600s cooldown, 5 max consec losses)
- `btcusd-live/ml_config.json` — ML config (H1, 365d, binary, 16 features, 60% confidence)
- `btcusd-live/ml/ensemble_predictor.py` — Ensemble voting (2/3 majority, binary)
- `btcusd-live/ml/ensemble_trainer.py` — Training with balanced downsampling + walk-forward
- `btcusd-live/ml/feature_engineering.py` — 16 features including hourly_return, daily_range_position
- `btcusd-live/demo_logger.py` — Structured CSV logging for demo validation
- `btcusd-live/backtest_ml.py` — ML-aware backtester with realistic position sizing
- `btcusd-live/logs/` — trades.csv, signals.csv, daily_summary.csv
- `btcusd-live/mt5_auth_live.json.bak` — LIVE credentials backup (DO NOT DELETE)

### Demo Bot (btcusd/ — active)
- `btcusd/trading.py` — Demo bot (~2500 lines). Has H1 momentum filter (Mar 6).
- `btcusd/config.json` — Daily limit 5, weekly limit 25, 5 max consec losses
- `btcusd/logs/` — trade_log.csv (review script source), signals.csv, daily_summary.csv, trades.csv (demo_logger detailed)
- `btcusd/state/` — state.json (persisted counters including H1 dedup timestamp)
- `btcusd/demo_logger.py` — Structured CSV logging

### Infrastructure
- `ecosystem.config.js` — PM2 config
- `auto_retrain.py` — Weekly model retraining
- `sync_to_supabase.py` — Trade data sync
- `daily_digest.py` — End-of-day summary

## Important Patterns

- **Binary Ensemble Voting:** `ensemble_predictor.py` — `most_common_count >= 2` for majority. HOLD = models disagree, not a predicted class.
- **Balanced Downsampling:** `_balance_training_data()` in `ensemble_trainer.py` — called before each model fit.
- **Session Trading:** Asian, EU, US sessions. Off-hours = +10% confidence threshold.
- **Circuit Breaker:** 5 consecutive losses → daily shutdown. STICKY: wins don't reset once triggered. $0.50 min win to reset counter (only if breaker hasn't fired).
- **State Persistence:** `save_state()` writes `state.json` after every trade close. `load_state()` on startup restores counters. Cross-checked with CSV restore (takes max). Resets at MYT midnight (daily) and Monday 00:00 MYT (weekly).
- **Dynamic SL/TP:** SL = 1.5× ATR, TP = 2.75× ATR (R:R = 1.83:1). Optimised by AutoResearch Mar 10.
- **Demo Logger:** `demo_logger.py` hooks into `log_blocked_signal()` and `log_trade_result()`.
- **Walk-Forward:** `walk_forward_validate()` in ensemble_trainer — 5 splits, rolling train/test.
- **H1 Candle Dedup:** `_last_evaluated_h1_candle` tracks last H1 candle open timestamp. Skips signal evaluation if candle hasn't changed. Prevents log spam and duplicate entries. **Persisted to state.json** (Mar 6) — survives PM2 restarts.
- **H1 Momentum Filter (Mar 6):** SELL requires lower close + lower high + close < EMA20. BUY requires higher close + higher low + close > EMA20. Blocked signals logged to signals.csv with reason `h1_momentum`.
- **Trailing Stop:** DISABLED (Feb 27). M1 ATR trailing was killing H1 trend trades in minutes.
- **Smart Exit:** DISABLED (Feb 27). Max hold 120min incompatible with multi-hour trend holds.
- **State Tracking (Mar 6):** `total_wins`/`total_losses` are cumulative — no daily reset. Daily counters (`daily_trade_count`, `daily_pl`, `consecutive_losses`, `circuit_breaker_triggered`) reset at MYT midnight.

## Config Quick Reference

### `btcusd/config.json` (Demo — active)
| Setting | Value |
|---------|-------|
| Timeframe | H1 |
| Lot Size | 0.01 |
| SL / TP | 1.5× / 2.75× ATR (R:R 1.83, optimised Mar 10) |
| H4 EMA Fast/Slow | 15 / 80 (optimised by AutoResearch Mar 8-10) |
| H1 Entry EMA | 25 (optimised from 20, Mar 8) |
| Trade Cooldown | 3600s (1 hour) |
| Circuit Breaker | 5 consecutive losses (⚠️ change to 3 for live) |
| Daily Limit | 5 trades |
| Weekly Limit | 30 trades (Mar 7) |
| ATR Floor | 250 on H1 (optimised from 300, Mar 8) |
| H1 Momentum Filter | Enabled (Mar 6) |

### `btcusd-live/config.json` (Live — stopped)
| Setting | Value |
|---------|-------|
| Timeframe | H1 |
| Lot Size | 0.01 |
| Trade Cooldown | 3600s (1 hour) |
| Circuit Breaker | 5 consecutive losses |
| Weekly Limit | 15 trades |
| EMA Trend Filter | Disabled |
| Momentum Filter | Disabled |

### `btcusd-live/ml_config.json`
| Setting | Value |
|---------|-------|
| Timeframe | H1 |
| Training Period | 365 days |
| Lookahead | 12 candles (12 hours) |
| TP / SL | 0.5% / 0.4% |
| Confidence | 60% |
| Min Prob Diff | 15% |
| Max Trades/Day | 3 |
| Features | 16 (binary, no HOLD class) |

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## AutoResearch System (Mar 8, 2026)

Karpathy-style autonomous parameter optimizer. AI agent proposes mutations, backtests on live MT5 data, keeps improvements, discards regressions.

### Files (all in `btcusd/`)
| File | Purpose |
|------|---------|
| `autotrader.py` | Main research loop — proposes mutations via Claude, backtests, keep/discard, Telegram approval gate |
| `backtest_mt5.py` | MT5 backtest engine — replays H4/H1 trend strategy against live historical data |
| `MEMORY.md` | Strategy memory — current params, baseline metrics, decision rules, mutable param ranges |
| `deploy.py` | Applies params to config/strategy, git push, PM2 stop+start. Auto-finds PM2 in PATH. Live verification disabled. |
| `telegram_gate.py` | Polls Telegram (@algotrade_mx_bot) for /deploy /skip /stop replies |
| `summary.py` | Morning report — shows all keeps/discards with param type column |
| `calibration.jsonl` | Experiment log (append-only) |

### How to Run
```bash
cd btcusd
python autotrader.py --hours 168 --delay 120    # overnight loop (~30 experiments/hour)
python autotrader.py --once --hours 168          # single experiment
python summary.py                                 # morning review
```

### Key Design
- **Model:** Claude Sonnet 4 (cost-efficient for JSON mutation proposals)
- **Backtest:** Real MT5 historical data (H4 trend + H1 signals), not synthetic
- **Dedup guard:** 4-hour cooldown between signals (matches live bot cooldown)
- **Dual drawdown:** Equity-peak DD (primary) + consecutive loss streak DD
- **Telegram gate:** /deploy, /skip, /stop from phone. 30-min timeout → auto-skip
- **Deploy pipeline:** Config update → git push → PM2 stop+start (bot loads new config on startup). Fully automatic, no human intervention.
- **Weekly cron:** `autotrader-weekly` in ecosystem.config.js — `0 23 * * 0` (Sunday 11PM MYT), 8h max runtime

### Mar 8 Results (206 experiments, 3% keep rate)
| Param | Before | After | Change |
|-------|--------|-------|--------|
| sl_atr_multiplier | 1.50 | **1.25** | Tighter SL |
| tp_atr_multiplier | 2.00 | **1.75** | Tighter TP |
| h4_ema_fast | 20 | **15** | Faster trend detection |
| h1_ema_period | 20 | **25** | Deeper pullback entries |
| min_atr | 300 | **250** | More trades in moderate vol |

Best: 54.3% WR, $82.75 PnL/week, 0.22% DD.

### Mar 10 Results (260+ total experiments, ~4% keep rate)
| Param | Mar 8 | Mar 10 | Change |
|-------|-------|--------|--------|
| sl_atr_multiplier | 1.25 | **1.5** | Wider SL (let trades breathe) |
| tp_atr_multiplier | 1.75 | **2.75** | Much wider TP (let winners run) |
| h4_ema_slow | 50 | **80** | More selective trend detection |

Best: **69.2% WR, $140.28 PnL/week, 0.06% DD, R:R 1.83.** Near convergence — all other param directions rejected.

## Task Management

1. **Plan First:** Write plan to `tasks/todo.md` with checkable items
2. **Verify Plans:** Check in before starting implementation
3. **Track Progress:** Mark items complete as you go
4. **Explain Changes:** High-level summary at each step
5. **Document Results:** Add review section to `tasks/todo.md`
6. **Capture Lessons:** Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First:** Make every change as simple as possible. Impact minimal code.
- **No Laziness:** Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact:** Changes should only touch what's necessary. Avoid introducing bugs.

## Development Guidelines

- **Branch:** `main` only. Auto-merge PRs granted.
- **Testing:** Restart PM2 after code changes: `pm2 restart bot-btcusd --update-env`
- **SECURITY — CRITICAL:** NEVER commit tokens/keys/secrets. This has caused GitHub alerts TWICE (Feb 19 + Feb 24). Rules:
  - ALWAYS use env vars or `.env` (gitignored) for credentials
  - NEVER use `git add -A` — always `git add <specific files>` and review staged files
  - One-off scripts with credentials belong in gitignored folders, not the repo
  - Telegram token in env var `TELEGRAM_BOT_TOKEN`, Supabase key in `.env`
- **Windows:** PowerShell syntax. ASCII-safe print (no emoji — cp1252 crashes).
- **Demo vs Live:** Demo runs from `btcusd/`, live codebase in `btcusd-live/` (untouched). `mt5_auth.json` controls which account. Live backup at `btcusd-live/mt5_auth_live.json.bak`.
- **Directory isolation:** ALL demo changes go in `btcusd/`. NEVER modify `btcusd-live/` during demo week.
