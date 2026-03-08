"""
backtest_mt5.py — MT5 Historical Backtest Engine
For use with autotrader.py (AutoTrader Research Loop)

Replays your TrendStrategy logic on historical H4/H1 candles.
No live orders placed — pure simulation using stored price data.

Usage (standalone):
  python backtest_mt5.py                         # test with current config.json params
  python backtest_mt5.py --hours 48              # extend backtest window
  python backtest_mt5.py --sl 1.5 --tp 2.5      # test specific ATR multipliers
  python backtest_mt5.py --h4-ema-fast 15        # test different H4 EMA periods

Called from autotrader.py:
  from backtest_mt5 import run_backtest
  result = run_backtest("btcusd", params, hours=24)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pandas as pd
import numpy as np

# ── MT5 import (graceful fallback for dry-run testing) ────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 not installed — only dry-run mode available")

# ── Default paths (assumes you run from btcusd/ directory) ───────────────────
AUTH_FILE   = Path("mt5_auth.json")
CONFIG_FILE = Path("config.json")
SYMBOL      = "BTCUSD"

# ── MT5 connection ─────────────────────────────────────────────────────────────

_mt5_connected = False

def connect_mt5() -> bool:
    """Initialize MT5 connection using mt5_auth.json credentials."""
    global _mt5_connected
    if _mt5_connected:
        return True
    if not MT5_AVAILABLE:
        return False
    if not AUTH_FILE.exists():
        raise FileNotFoundError(f"{AUTH_FILE} not found")

    with open(AUTH_FILE) as f:
        auth = json.load(f)

    if not mt5.initialize(
        login=int(auth["login"]),
        password=auth["password"],
        server=auth["server"],
    ):
        print(f"[MT5] initialize() failed: {mt5.last_error()}")
        return False

    _mt5_connected = True
    print(f"[MT5] Connected — {mt5.account_info().company} | "
          f"balance={mt5.account_info().balance:.2f}")
    return True


def disconnect_mt5():
    global _mt5_connected
    if MT5_AVAILABLE and _mt5_connected:
        mt5.shutdown()
        _mt5_connected = False


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_candles(symbol: str, timeframe_str: str, hours: int) -> pd.DataFrame:
    """
    Fetch historical OHLCV candles from MT5.
    Returns DataFrame with columns: time, open, high, low, close, tick_volume
    """
    tf_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    tf = tf_map.get(timeframe_str.upper(), mt5.TIMEFRAME_H1)

    # Fetch extra candles for indicator warmup
    warmup    = 200
    now       = datetime.now(UTC)
    from_time = now - timedelta(hours=hours + warmup * 4)  # rough buffer

    rates = mt5.copy_rates_range(symbol, tf, from_time, now)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"[MT5] No data for {symbol} {timeframe_str}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ── Indicator helpers ──────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# ── Strategy signal replay ─────────────────────────────────────────────────────

def compute_h4_trend(h4_df: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.Series:
    """
    Replicate TrendStrategy.get_trend_direction() across all H4 candles.
    Returns Series: 'bullish' | 'bearish' | 'neutral' indexed by time.
    """
    h4 = h4_df.copy()
    h4["ema_fast"] = calc_ema(h4["close"], ema_fast)
    h4["ema_slow"] = calc_ema(h4["close"], ema_slow)

    def _trend(row):
        if row["close"] > row["ema_fast"] > row["ema_slow"]:
            return "bullish"
        elif row["close"] < row["ema_fast"] < row["ema_slow"]:
            return "bearish"
        return "neutral"

    h4["trend"] = h4.apply(_trend, axis=1)
    return h4.set_index("time")["trend"]


def replay_signals(
    h1_df: pd.DataFrame,
    h4_trend_series: pd.Series,
    h1_ema_period: int,
    atr_period: int,
    min_atr: float,
) -> pd.DataFrame:
    """
    Replay H1 entry signal logic on historical candles.
    Replicates TrendStrategy.get_h1_entry_signal() for each H1 bar.

    Returns DataFrame of signal events with columns:
      time, signal (buy/sell), entry_price, trend, atr, reason
    """
    h1 = h1_df.copy()
    h1["ema20"]  = calc_ema(h1["close"], h1_ema_period)
    h1["ema50"]  = calc_ema(h1["close"], 50)
    h1["atr"]    = calc_atr(h1, atr_period)

    signals = []

    for i in range(2, len(h1)):
        row  = h1.iloc[i]
        prev = h1.iloc[i - 1]

        # Map H1 bar to H4 trend (find latest H4 bar before this H1 bar)
        h1_time   = row["time"]
        h4_before = h4_trend_series[h4_trend_series.index <= h1_time]
        if h4_before.empty:
            continue
        trend = h4_before.iloc[-1]

        # ATR filter
        if pd.isna(row["atr"]) or row["atr"] < min_atr:
            continue

        price = row["close"]
        ema20 = row["ema20"]
        low   = row["low"]
        high  = row["high"]
        atr   = row["atr"]
        signal = None
        reason = ""

        if trend == "bullish":
            if low <= ema20 * 1.001 and price > ema20:
                signal = "buy"
                reason = f"pullback to EMA{h1_ema_period}"
            elif price > prev["high"] and price > ema20:
                signal = "buy"
                reason = "H1 breakout"

        elif trend == "bearish":
            if high >= ema20 * 0.999 and price < ema20:
                signal = "sell"
                reason = f"pullback to EMA{h1_ema_period}"
            elif price < prev["low"] and price < ema20:
                signal = "sell"
                reason = "H1 breakdown"

        if signal:
            signals.append({
                "time":        h1_time,
                "bar_index":   i,
                "signal":      signal,
                "entry_price": price,
                "trend":       trend,
                "atr":         atr,
                "reason":      reason,
            })

    return pd.DataFrame(signals)


# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate_trades(
    signals_df:       pd.DataFrame,
    h1_df:            pd.DataFrame,
    sl_atr_mult:      float,
    tp_atr_mult:      float,
    lot_size:         float = 0.01,
    max_hold_candles: int   = 48,
    account_balance:  float = 10000.0,  # used for drawdown % denominator
) -> dict:
    """
    Simulate trade outcomes for each signal.

    PnL: BTCUSD 0.01 lot × ATR move in USD = direct P&L
         e.g. 0.01 × $1050 ATR × 1.5 SL_mult = -$15.75 on a loss

    Drawdown: max_consecutive_loss_usd / account_balance * 100
      This is the only stable metric when equity starts at 0.
      Equity-peak-based % is meaningless when early trades are losses.
    """
    if signals_df.empty:
        return {
            "win_rate":         0.0,
            "pnl":              0.0,
            "drawdown":         0.0,
            "drawdown_usd":     0.0,
            "total_trades":     0,
            "wins":             0,
            "losses":           0,
            "avg_hold_candles": 0,
        }

    results   = []
    hold_list = []

    # Track worst drawdown as largest cumulative loss streak (USD)
    running_loss_usd = 0.0
    max_loss_streak  = 0.0

    for _, sig in signals_df.iterrows():
        bar_idx   = sig["bar_index"]
        entry     = sig["entry_price"]
        atr       = sig["atr"]
        direction = sig["signal"]

        sl_dist = atr * sl_atr_mult
        tp_dist = atr * tp_atr_mult

        if direction == "buy":
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        outcome  = "timeout"
        hold     = 0
        exit_pnl = 0.0

        for j in range(bar_idx + 1, min(bar_idx + max_hold_candles + 1, len(h1_df))):
            candle = h1_df.iloc[j]
            hold  += 1
            if direction == "buy":
                if candle["low"] <= sl_price:
                    outcome  = "loss"
                    exit_pnl = -(sl_dist * lot_size)
                    break
                if candle["high"] >= tp_price:
                    outcome  = "win"
                    exit_pnl = tp_dist * lot_size
                    break
            else:
                if candle["high"] >= sl_price:
                    outcome  = "loss"
                    exit_pnl = -(sl_dist * lot_size)
                    break
                if candle["low"] <= tp_price:
                    outcome  = "win"
                    exit_pnl = tp_dist * lot_size
                    break

        if outcome == "timeout":
            exit_pnl = -(sl_dist * 0.3 * lot_size)

        results.append({"direction": direction, "outcome": outcome,
                        "pnl": exit_pnl, "hold": hold})
        hold_list.append(hold)

        # Track consecutive loss streak for drawdown
        if exit_pnl < 0:
            running_loss_usd += abs(exit_pnl)
            max_loss_streak   = max(max_loss_streak, running_loss_usd)
        else:
            running_loss_usd = 0.0  # reset on any win

    df_res    = pd.DataFrame(results)
    wins      = int((df_res["outcome"] == "win").sum())
    losses    = int((df_res["outcome"] != "win").sum())
    total     = len(df_res)
    win_rate  = round(float(wins / total * 100) if total > 0 else 0.0, 1)
    total_pnl = round(float(df_res["pnl"].sum()), 2)
    avg_hold  = round(float(sum(hold_list) / len(hold_list)) if hold_list else 0.0, 1)

    # Drawdown % = worst loss streak as % of account balance
    dd_pct = round(max_loss_streak / account_balance * 100, 2)

    return {
        "win_rate":          win_rate,
        "pnl":               total_pnl,
        "drawdown":          dd_pct,
        "drawdown_usd":      round(max_loss_streak, 2),
        "total_trades":      total,
        "wins":              wins,
        "losses":            losses,
        "avg_hold_candles":  avg_hold,
    }


# ── Public API: run_backtest() ─────────────────────────────────────────────────

def run_backtest(
    strategy_name: str,
    params:        dict,
    hours:         int  = 24,
    dry_run:       bool = False,
) -> dict:
    """
    Main entry point called by autotrader.py.

    params dict (agent provides these — all optional, falls back to config.json):
      sl_atr_multiplier  : float  (default 1.5)
      tp_atr_multiplier  : float  (default 2.0)
      atr_period         : int    (default 14)
      min_atr            : float  (default 300)
      h4_ema_fast        : int    (default 20)
      h4_ema_slow        : int    (default 50)
      h1_ema_period      : int    (default 20)
      lot_size           : float  (default 0.01)
      max_hold_candles   : int    (default 48)

    Returns:
      {win_rate, pnl, drawdown, total_trades, duration_s}
    """
    t0 = time.time()

    # ── Load config fallbacks ──────────────────────────────────────────────────
    cfg = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)

    dyn = cfg.get("dynamic_sltp", {})
    sl_mult      = params.get("sl_atr_multiplier", dyn.get("sl_atr_multiplier", 1.5))
    tp_mult      = params.get("tp_atr_multiplier", dyn.get("tp_atr_multiplier", 2.0))
    atr_period   = int(params.get("atr_period",    cfg.get("atr_period",  14)))
    min_atr      = float(params.get("min_atr",     cfg.get("min_atr",    300)))
    h4_ema_fast  = int(params.get("h4_ema_fast",   20))
    h4_ema_slow  = int(params.get("h4_ema_slow",   50))
    h1_ema_period= int(params.get("h1_ema_period", 20))
    lot_size     = float(params.get("lot_size",    0.01))
    max_hold     = int(params.get("max_hold_candles", 48))
    symbol       = cfg.get("symbol", SYMBOL)

    # ── Dry run ────────────────────────────────────────────────────────────────
    if dry_run:
        import random
        time.sleep(random.uniform(1.0, 3.0))
        base_wr = 54.0
        noise   = random.uniform(-5.0, 8.0)
        wr      = round(base_wr + noise, 1)
        pnl     = round(random.uniform(-10, 30), 2)
        dd      = round(random.uniform(1.5, 7.0), 2)
        total   = random.randint(5, 25)
        wins    = round(total * wr / 100)
        losses  = total - wins
        return {
            "win_rate":         wr,
            "pnl":              pnl,
            "drawdown":         dd,
            "total_trades":     total,
            "wins":             wins,
            "losses":           losses,
            "avg_hold_candles": random.randint(3, 15),
            "duration_s":       round(time.time() - t0, 2),
        }

    # ── Live MT5 backtest ──────────────────────────────────────────────────────
    if not connect_mt5():
        raise RuntimeError("MT5 connection failed. Check mt5_auth.json and MT5 terminal.")

    # Fetch H1 with 200 extra candles for EMA/ATR warmup
    h1_warmup_hours = hours + 200
    print(f"  [BT] Fetching data for {symbol} (H4: {hours+200}h, H1 with warmup: {h1_warmup_hours}h)...")
    h4_df   = fetch_candles(symbol, "H4", hours + 200)
    h1_full = fetch_candles(symbol, "H1", h1_warmup_hours)

    cutoff   = datetime.now(UTC) - timedelta(hours=hours)
    h1_window = h1_full[h1_full["time"] >= cutoff]
    print(f"  [BT] H4 total: {len(h4_df)} | H1 total (warmup+window): {len(h1_full)} | H1 in window: {len(h1_window)}")

    # ── DEBUG: H4 trend distribution ─────────────────────────────────────────
    h4_trend  = compute_h4_trend(h4_df, h4_ema_fast, h4_ema_slow)
    h4_recent = h4_trend[h4_trend.index >= cutoff]
    trend_counts = h4_recent.value_counts().to_dict()
    print(f"  [BT] H4 trend (last {hours}h): {trend_counts}")

    # ── DEBUG: ATR vs min_atr ─────────────────────────────────────────────────
    _atr_series = calc_atr(h1_full, atr_period)
    atr_recent  = _atr_series[h1_full["time"] >= cutoff]
    if not atr_recent.dropna().empty:
        pct_above = (atr_recent.dropna() >= min_atr).mean() * 100
        print(f"  [BT] H1 ATR (last {hours}h): min={atr_recent.min():.0f}  avg={atr_recent.mean():.0f}  max={atr_recent.max():.0f}  threshold={min_atr}  above={pct_above:.0f}%")

    # ── DEBUG: last 3 H1 bars vs EMA20 ───────────────────────────────────────
    _ema20_series = calc_ema(h1_full["close"], h1_ema_period)
    for idx in h1_window.tail(3).index:
        r = h1_full.loc[idx]
        ema20_val = _ema20_series.loc[idx]
        gap_pct = (r["close"] - ema20_val) / ema20_val * 100
        # find h4 trend for this bar
        h4_for_bar = h4_trend[h4_trend.index <= r["time"]]
        bar_trend = h4_for_bar.iloc[-1] if not h4_for_bar.empty else "n/a"
        print(f"  [BT]   {r['time'].strftime('%m-%d %H:%M')} close={r['close']:.0f} ema20={ema20_val:.0f} low={r['low']:.0f} high={r['high']:.0f} gap={gap_pct:+.2f}% trend={bar_trend}")

    # ── Run signal replay ─────────────────────────────────────────────────────
    signals = replay_signals(
        h1_df           = h1_full,
        h4_trend_series = h4_trend,
        h1_ema_period   = h1_ema_period,
        atr_period      = atr_period,
        min_atr         = min_atr,
    )

    # Filter signals to backtest window only
    if not signals.empty:
        signals = signals[signals["time"] >= cutoff].reset_index(drop=True)

    print(f"  [BT] Signals found in window: {len(signals)}")

    # Get account balance for drawdown % calculation
    account_bal = 10000.0
    try:
        info = mt5.account_info()
        if info:
            account_bal = float(info.balance)
    except Exception:
        pass

    # Simulate trades
    metrics = simulate_trades(
        signals_df       = signals,
        h1_df            = h1_full,
        sl_atr_mult      = sl_mult,
        tp_atr_mult      = tp_mult,
        lot_size         = lot_size,
        max_hold_candles = max_hold,
        account_balance  = account_bal,
    )

    metrics["duration_s"] = round(time.time() - t0, 1)

    print(f"  [BT] Result: {metrics}")
    return metrics


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 Trend Strategy Backtest")
    parser.add_argument("--hours",      type=int,   default=168,  help="Backtest window in hours")
    parser.add_argument("--sl",         type=float, default=1.5,  help="SL ATR multiplier")
    parser.add_argument("--tp",         type=float, default=2.0,  help="TP ATR multiplier")
    parser.add_argument("--atr-period", type=int,   default=14,   help="ATR period")
    parser.add_argument("--min-atr",    type=float, default=300,  help="Minimum ATR filter")
    parser.add_argument("--h4-ema-fast",type=int,   default=20,   help="H4 EMA fast period")
    parser.add_argument("--h4-ema-slow",type=int,   default=50,   help="H4 EMA slow period")
    parser.add_argument("--h1-ema",     type=int,   default=20,   help="H1 EMA entry period")
    parser.add_argument("--lot",        type=float, default=0.01, help="Lot size")
    parser.add_argument("--dry-run",    action="store_true",      help="Simulate without MT5")
    args = parser.parse_args()

    params = {
        "sl_atr_multiplier": args.sl,
        "tp_atr_multiplier": args.tp,
        "atr_period":        args.atr_period,
        "min_atr":           args.min_atr,
        "h4_ema_fast":       args.h4_ema_fast,
        "h4_ema_slow":       args.h4_ema_slow,
        "h1_ema_period":     args.h1_ema,
        "lot_size":          args.lot,
    }

    print("=" * 55)
    print("  MT5 Trend Strategy Backtest")
    print(f"  Window: {args.hours}h | SL: {args.sl}x | TP: {args.tp}x")
    print(f"  H4 EMA: {args.h4_ema_fast}/{args.h4_ema_slow} | H1 EMA: {args.h1_ema}")
    print(f"  Min ATR: {args.min_atr} | Lot: {args.lot}")
    print("=" * 55)

    try:
        result = run_backtest("btcusd", params, hours=args.hours, dry_run=args.dry_run)
        print("\n── Final Result ──────────────────────────────────────")
        print(f"  Win rate    : {result['win_rate']}%")
        print(f"  Total PnL   : ${result['pnl']}")
        print(f"  Max drawdown: {result['drawdown']}% (${result.get('drawdown_usd', 0):.2f})")
        print(f"  Trades      : {result['total_trades']} "
              f"(W:{result.get('wins',0)} / L:{result.get('losses',0)})")
        print(f"  Avg hold    : {result.get('avg_hold_candles',0)} H1 candles")
        print(f"  Duration    : {result['duration_s']}s")
    finally:
        disconnect_mt5()
