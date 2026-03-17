# MT5 Bot — Demo Week 3 Fix Instructions
**From:** Claude (code review layer)  
**To:** Acu  
**Date:** 2026-03-06  
**Priority:** Fix all issues below before starting Demo Week 3

---

## ⚠️ CRITICAL — Folder Safety Rule

> **ALL changes in this document apply ONLY to the `btcusd` folder (demo bot).**  
> **NEVER touch the `btcusd-live` folder under any circumstance.**

```
✅  btcusd/          ← ALL fixes go here (demo bot)
❌  btcusd-live/     ← DO NOT TOUCH (live bot)
```

Before making any file edit, confirm the working path starts with `btcusd/` and not `btcusd-live/`. If there is any doubt, stop and ask for confirmation first.

---

## Context

Demo Week 2 review is complete. The bot ran 15/15 trades and went idle. Results:

| Metric | Result | Target | Status |
|---|---|---|---|
| Win Rate | 26.7% | >45% | ❌ FAIL |
| Profit Factor | 0.34 | >1.1 | ❌ FAIL |
| Net PnL | -$51.48 | Positive | ❌ FAIL |
| BUY Net PnL | -$3.53 | Positive | ⚠️ Close |
| SELL Net PnL | -$47.95 | Positive | ❌ Critical |

**Do NOT proceed to live. Run Demo Week 3 after all fixes below.**

---

## Fix #0 — Consolidate All Log Files to `/logs/` Folder
**Priority: CRITICAL (do this first before any other fix)**

### Problem
Log files are currently scattered. All output files must write to one consistent location so reviews, monitoring, and debugging all pull from the same place.

### Required folder structure
```
project_root/
├── logs/
│   ├── trade_log.csv
│   ├── signals.csv
│   └── daily_summary.csv
├── state/
│   └── state.json
└── bot.py (or equivalent entry point)
```

> **Note:** `state.json` goes in `state/` not `logs/` — it is a live runtime file, not a log. Mixing it with logs risks accidental overwrite during log rotation.

### Fix
Update all file write paths in the bot code:

```python
import os

# Define paths once at the top of your config/constants file
LOGS_DIR  = os.path.join(os.path.dirname(__file__), "logs")
STATE_DIR = os.path.join(os.path.dirname(__file__), "state")

# Ensure folders exist on startup
os.makedirs(LOGS_DIR,  exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# File paths
TRADE_LOG_PATH     = os.path.join(LOGS_DIR,  "trade_log.csv")
SIGNALS_LOG_PATH   = os.path.join(LOGS_DIR,  "signals.csv")
DAILY_SUMMARY_PATH = os.path.join(LOGS_DIR,  "daily_summary.csv")
STATE_PATH         = os.path.join(STATE_DIR, "state.json")
```

Replace all hardcoded file paths in the bot with these constants.

### Verify
After fix: confirm all 3 CSV files appear in `logs/` and `state.json` appears in `state/` after the bot runs for 5 minutes.

---

## Fix #1 — State Tracking: `total_wins` / `total_losses` Not Updating
**Priority: CRITICAL (safety mechanism is blind)**

### Problem
`state.json` shows `total_wins: 0` and `total_losses: 0` despite 15 real trades executing (4 wins, 11 losses). The circuit breaker depends on these counters — with zeros it has no real data to act on.

`daily_summary.csv` also shows `trades_taken: 0` every single day despite real trades executing.

### Fix
After every trade closes, increment the correct counter in state:

```python
# After trade closes
if trade_pnl > 0:
    state["total_wins"] += 1
else:
    state["total_losses"] += 1

# Also update daily_summary for the current date
daily_summary[today]["trades_taken"] += 1
daily_summary[today]["wins" if trade_pnl > 0 else "losses"] += 1
daily_summary[today]["daily_pnl"] += trade_pnl

# Save state immediately after update
save_state(state, STATE_PATH)
```

### Verify
Run one manual test trade, close it, then check:
- `state/state.json` → `total_wins` or `total_losses` incremented by 1
- `logs/daily_summary.csv` → today's row shows `trades_taken: 1`

---

## Fix #2 — Signal Deduplication: 127 Signals Fired in One Day
**Priority: HIGH (wastes trade budget, distorts all metrics)**

### Problem
`logs/signals.csv` shows 127 BUY signals on 2026-02-27 alone — one signal per minute. The signal loop runs on a 1-minute tick without checking "have I already signalled this H1 setup?"

After analysis, realistic signal count after dedup:

| Date | Raw Signals | After Dedup |
|---|---|---|
| 2026-02-26 | 9 | 1 |
| 2026-02-27 | 127 | 5 |
| 2026-02-28 | 4 | 4 |
| 2026-03-01 | 5 | 5 |
| 2026-03-02 | 8 | 8 |
| 2026-03-03 | 7 | 7 |
| 2026-03-04 | 11 | 11 |
| **Average** | **~22/day** | **~6/day** |

### Fix
Add an H1 candle cooldown gate. Only fire one signal per H1 candle per direction:

```python
# Track last signal candle time per direction
last_signal_candle = {"buy": None, "sell": None}

def should_fire_signal(direction, current_h1_candle_open_time):
    """Only fire if we haven't already fired on the current H1 candle."""
    last = last_signal_candle.get(direction)
    if last == current_h1_candle_open_time:
        return False  # Already fired this candle
    return True

# After signal fires successfully:
last_signal_candle[direction] = current_h1_candle_open_time
```

### Verify
Check `logs/signals.csv` after fix — no more than 1 signal per hour per direction.

---

## Fix #3 — SELL Entry Filter: All 6 SELL Losses Were Counter-Momentum
**Priority: HIGH (responsible for 93% of total losses)**

### Problem
SELL performance breakdown:

| Date | Entry | Exit | PnL | H1 Reality |
|---|---|---|---|---|
| 2026-02-28 | 65,567 | 64,704 | +$8.64 | ✅ H1 dropping |
| 2026-02-28 | 63,749 | 64,494 | -$7.45 | ❌ H1 bouncing |
| 2026-02-28 | 64,841 | 65,883 | -$10.43 | ❌ H1 rising |
| 2026-03-01 | 66,319 | 67,088 | -$7.69 | ❌ H1 rising |
| 2026-03-02 | 66,023 | 66,929 | -$9.06 | ❌ H1 rising |
| 2026-03-02 | 65,997 | 67,160 | -$11.62 | ❌ H1 rising |
| 2026-03-02 | 67,158 | 68,193 | -$10.34 | ❌ H1 rising |

H4 said bearish but H1 was making higher highs. Bot entered SELL into a rising H1 market.

### Fix
Add H1 momentum confirmation before entry. Only enter SELL if H1 confirms bearish momentum:

```python
def is_valid_sell_entry(h1_candles):
    """
    Require H1 to confirm bearish momentum before entering SELL.
    All 3 conditions must be true:
      1. Current H1 close < previous H1 close  (making lower closes)
      2. Current H1 high  < previous H1 high   (making lower highs)
      3. Current H1 close < H1 EMA20           (price below short-term average)
    """
    curr = h1_candles[-1]
    prev = h1_candles[-2]

    return (
        curr["close"] < prev["close"] and
        curr["high"]  < prev["high"]  and
        curr["close"] < curr["ema20"]
    )

def is_valid_buy_entry(h1_candles):
    """Mirror confirmation for BUY entries."""
    curr = h1_candles[-1]
    prev = h1_candles[-2]

    return (
        curr["close"] > prev["close"] and
        curr["low"]   > prev["low"]   and
        curr["close"] > curr["ema20"]
    )
```

Apply both filters before any trade entry.

---

## Fix #4 — Trade Limits: Keep Them, But Adjust the Numbers
**Priority: MEDIUM**

### Current state
- Weekly limit: 15 trades → hit by Day 3, bot went idle rest of week
- Root cause: signal spam bug (Fix #2) consumed the budget, not real trade opportunities

### Recommendation: Keep limits, adjust numbers

> ⚠️ **Do NOT remove limits entirely.** Limits are your last line of defence if signal logic has a bug. Without them, a broken bot can open hundreds of trades in minutes on a live account.

After Fix #2 (dedup), realistic signal flow is **~6 signals/day**, **~30/week**. Not every signal becomes a trade — the H1 confirmation filter will block some. A reasonable execution rate is 40–60% of signals.

### New recommended limits

| Limit | Old Value | New Value | Reasoning |
|---|---|---|---|
| Daily trade limit | unknown | **5 trades/day** | ~6 signals/day after dedup, confirmation filter blocks ~1-2 |
| Weekly trade limit | 15 | **25 trades/week** | 5/day × 5 days, with buffer for active weeks |

### Fix
Update limits in your config/constants:

```python
MAX_TRADES_PER_DAY  = 5   # was: unknown / not set
MAX_TRADES_PER_WEEK = 25  # was: 15
```

> The daily limit (5/day) is the real guard. The weekly limit (25) is a backstop. If the bot hits 5 trades in one day, that warrants a manual review — it likely means the filter is too loose.

---

## Fix #5 — Circuit Breaker Key Mismatch in Review Script
**Priority: LOW**

In `mt5_review.py`, update the `cb_keys` list to include the actual key name used in `state.json`:

```python
cb_keys = [
    'circuit_breaker',
    'circuit_breaker_active',
    'circuit_breaker_triggered',   # ← add this (matches state.json)
    'cb_triggered',
    'daily_loss',
    'weekly_loss',
    'is_halted',
    'trading_halted'
]
```

---

## Full Checklist for Acu

| # | Fix | Files to change | Priority |
|---|---|---|---|
| 0 | Consolidate all logs to `logs/` folder, state to `state/` | All file write paths in bot | 🔴 Critical |
| 1 | Fix `total_wins`/`total_losses` on trade close | State manager / trade close handler | 🔴 Critical |
| 2 | Add H1 candle dedup gate to signal loop | Signal generator | 🟠 High |
| 3 | Add H1 momentum confirmation for SELL + BUY entry | Entry logic | 🟠 High |
| 4 | Change daily limit to 5, weekly limit to 25 | Config / constants | 🟡 Medium |
| 5 | Add `circuit_breaker_triggered` to review script | `mt5_review.py` | 🟢 Low |

**All done → deploy to demo → run Demo Week 3 → re-run `mt5_review.py` → share results with Claude.**

---

## Demo Week 3 Success Targets

| Metric | Minimum to pass | Notes |
|---|---|---|
| Win Rate | ≥ 45% | Overall |
| Profit Factor | ≥ 1.1 | Overall |
| SELL Win Rate | ≥ 35% | Was 14% in Week 2 |
| Signals per day | ≤ 8 | After dedup fix |
| Net PnL | Positive | Any positive amount |
| `state.json` total_wins | Matches actual winners | Data integrity check |
| Trades per day | ≤ 5 | New daily limit |
