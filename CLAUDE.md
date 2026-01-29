# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based automated trading bot for MetaTrader 5 (MT5) that supports multiple trading strategies with comprehensive risk management, backtesting, and notification systems.

## Key Commands

### Running the Bot
```bash
python trading.py
```
Main trading bot that runs continuously, monitors markets, executes trades, and manages positions.

### Running Backtests
```bash
python backtest.py
```
Optimizes strategy parameters across all available strategies (MA crossover, RSI, MACD, Bollinger Bands) and saves results to `strategy_params.json`. Automatically runs every Monday 8:00-12:00 AM US Eastern when the main bot is active.

### Dependencies
```bash
pip install MetaTrader5 pandas numpy requests pytz
```

## Architecture

### Core Files
- **trading.py**: Main trading loop with strategy execution, position management, trailing stops, and notification system
- **backtest.py**: Strategy optimization engine that tests multiple parameter combinations and saves best-performing parameters
- **config.json**: User-configurable trading parameters (stop loss, take profit, strategy selection, timeframes, risk limits)
- **mt5_auth.json**: Credentials for MT5 connection and Telegram API (excluded from git)
- **strategy_params.json**: Auto-generated file storing optimized parameters for each strategy
- **trade_log.csv**: Persistent log of all closed trades with P/L
- **trade_notifications.log**: All bot events, trades, errors, and alerts

### Strategy System

The bot supports 4 trading strategies selected via `config.json`:

1. **ma_crossover**: Moving average crossover using optimized short/long MA periods
2. **rsi**: Relative Strength Index with optimized period
3. **macd**: MACD indicator with optimized fast/slow/signal periods
4. **bollinger**: Bollinger Bands with optimized period and standard deviation

Strategy parameters are:
- Initially loaded from `strategy_params.json` (or defaults from `config.json`)
- Re-optimized weekly via `backtest.py` based on historical data
- Hot-reloaded into `trading.py` after optimization completes

### Trading Flow

1. **Initialization**: Connect to MT5, load config and optimized parameters
2. **Market Check**: Verify symbol availability and trading hours
3. **Signal Generation**:
   - Calculate indicators based on current strategy
   - Check multi-timeframe trend confirmation (lower + higher timeframe alignment)
   - Apply ATR volatility filter (min_atr threshold)
   - Apply news event filter (placeholder for news API integration)
4. **Position Management**:
   - Open new positions when all filters pass
   - Reverse positions when signal changes
   - Update trailing stops based on ATR
   - Log all trades to CSV and send Telegram notifications
5. **Risk Management**:
   - Monitor daily P/L against max_daily_loss and max_daily_profit
   - Pause trading and send alerts when limits reached
   - Send daily summaries at 5pm ET
   - Send weekly summaries on Fridays after 5pm ET

### Multi-Timeframe Filter

All strategies require confirmation from both:
- **Primary timeframe** (config.json: `timeframe`, e.g., M5)
- **Higher timeframe** (config.json: `higher_timeframe`, e.g., H1)

The `get_higher_tf_trend()` function calculates MA crossover on the higher timeframe and blocks trades that don't align with the higher timeframe trend direction.

### ATR-Based Features

- **Volatility Filter**: Blocks trades when ATR is below `min_atr` threshold (set to 0 to disable)
- **Trailing Stop**: When `enable_trailing_stop` is true, dynamically adjusts stop loss based on ATR * `atr_multiplier`

### Notification System

The `log_notify()` function handles all logging:
1. Prints to console
2. Writes to `trade_notifications.log` with UTC timestamp
3. Sends to Telegram via Bot API

Message suppression prevents spam from repeated filter messages.

### Automated Weekly Optimization

Every Monday 8:00-12:00 AM US Eastern, `trading.py`:
1. Spawns `backtest.py` as subprocess
2. Waits for completion
3. Reloads `strategy_params.json`
4. Continues trading with updated parameters

If backtesting fails, the bot continues with previous parameters (safe fallback).

## Configuration

### Critical Settings in config.json

- `strategy`: Which strategy to use (ma_crossover, rsi, macd, bollinger)
- `timeframe`: Primary trading timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
- `higher_timeframe`: Higher timeframe for trend confirmation
- `sl_pips`, `tp_pips`: Stop loss and take profit in pips
- `max_daily_loss`, `max_daily_profit`: Daily risk limits in account currency
- `min_atr`: Minimum ATR for volatility filter (0 disables)
- `atr_period`, `atr_multiplier`: ATR calculation and trailing stop settings
- `enable_trailing_stop`: Enable/disable ATR-based trailing stops
- `news_block_minutes`: Minutes before/after news to block trading (requires API integration)
- `backtest_period_days`: Historical data period for optimization (default 365)
- `forward_test_period_days`: Recent data period for forward testing (default 28)

### Credentials in mt5_auth.json

```json
{
  "login": 123456789,
  "password": "YOUR_MT5_PASSWORD",
  "server": "YOUR_MT5_SERVER",
  "telegram": {
    "bot_name": "@your_telegram_bot",
    "api_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "chat_id": 123456789
  }
}
```

## Implementation Notes

### Adding New Strategies

1. Add strategy parameters to `config.json` (or rely on `strategy_params.json`)
2. In `trading.py`: Add strategy logic in the strategy selection block (around line 269-306)
3. In `backtest.py`: Add optimization logic in `run_all_strategies_backtest()` and forward test in `run_all_strategies_forward_test()`
4. Update `strategy_params.json` structure to include new strategy key

### Extending News Filter

The `is_high_impact_news_near()` function (trading.py:138-142) is a placeholder. To integrate real news filtering:
1. Choose a news API (Forex Factory, Financial Modeling Prep, etc.)
2. Fetch upcoming high-impact events for the symbol
3. Return True if current time is within `news_block_minutes` of an event
4. Handle API errors gracefully (default to False to allow trading)

### Multi-Symbol Trading

Current implementation is single-symbol (EURUSD hardcoded). To support multiple symbols:
1. Refactor main loop to iterate over a list of symbols
2. Maintain separate position tracking per symbol
3. Ensure daily P/L aggregates across all symbols
4. Consider symbol-specific config sections

### Timezone Handling

The bot uses:
- `datetime.now(UTC)` for all MT5 operations and internal timestamps
- `ZoneInfo("America/New_York")` for scheduling (backtest automation, summaries, alerts)
- Fallback to UTC-4 offset if zoneinfo is unavailable

### Trade Logging

Two logging mechanisms:
1. **In-memory `trade_log` list**: Reset daily, used for summaries
2. **Persistent `trade_log.csv`**: Appended on every trade close, format: [timestamp, direction, entry_price, exit_price, profit]

## Safety and Testing

- **Never commit `mt5_auth.json`** - contains sensitive credentials
- Always test with demo account before live trading
- Monitor Telegram notifications for critical alerts and daily summaries
- The bot pauses trading (1 hour sleep) when max daily loss/profit is reached
- All trades are logged with timestamps for audit trail

---

## Changelog

### 2026-01-29: ML Threshold Optimization
**Problem:** Bot made 0 trades in 41 hours despite running continuously.

**Root Cause:** ML confidence threshold (65%) was too high for a 3-class model. Actual model output was 29-52%, never reaching the threshold.

**Changes to `ml_config.json`:**
| Setting | Before | After | Rationale |
|---------|--------|-------|-----------|
| `confidence_threshold` | 0.65 | 0.50 | With 3-class model (buy/sell/hold), 50% is realistic for volatile BTC markets |
| `min_probability_diff` | 0.15 | 0.12 | Maintain edge without being overly strict |
| `max_hold_probability` | 0.55 | 0.50 | Trade when HOLD signal is uncertain |

**Expected Result:** ~3-8 trades per day with reasonable signal quality.

**Branch:** `enhance-ml-thresholds` → merged to `main`
