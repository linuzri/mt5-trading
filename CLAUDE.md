# CLAUDE.md - AI Agent Context

This file provides context for AI agents working on this codebase.

## Project Overview

Automated MT5 trading bot. **Path B architecture**: rule-based trend-following (H4 direction + H1 pullback/breakout entries) with ATR-based risk management. ML ensemble trained but dormant (quality filter, not in trading loop).

### Current Status (Feb 27, 2026)
- **LIVE:** Account 51439249 — **STOPPED** (pending demo validation)
- **DEMO:** Account 61459537 — Running from `btcusd/` (PM2: `bot-btcusd`)
- **Strategy:** `trend_following` — H4 EMA20/50 alignment + H1 pullback/breakout entries
- **Demo Week 2:** March 2-7 (bot running over weekend for testing)
- **ML Model:** Trained (trade quality filter), loaded but NOT in trading loop
- **MQL5 Signal:** https://www.mql5.com/en/signals/2359964 — LIVE, APPROVED ✅
- **Auto-merge PRs:** Granted — merge directly without review

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

### Go-Live Criteria (Demo Week 2, review March 7)
- Win rate > 45%
- Profit factor > 1.1
- BUY/SELL direction follows H4 trend
- No single-day drawdown > 10% of account
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
- **Dynamic SL/TP:** SL = 1.5× ATR, TP = 2.0× ATR (R:R = 1.33:1).
- **Demo Logger:** `demo_logger.py` hooks into `log_blocked_signal()` and `log_trade_result()`.
- **Walk-Forward:** `walk_forward_validate()` in ensemble_trainer — 5 splits, rolling train/test.
- **H1 Candle Dedup:** `_last_evaluated_h1_candle` tracks last H1 candle open timestamp. Skips signal evaluation if candle hasn't changed. Prevents log spam and duplicate entries.
- **Trailing Stop:** DISABLED (Feb 27). M1 ATR trailing was killing H1 trend trades in minutes.
- **Smart Exit:** DISABLED (Feb 27). Max hold 120min incompatible with multi-hour trend holds.

## Config Quick Reference

### `btcusd-live/config.json`
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
