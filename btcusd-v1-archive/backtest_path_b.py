#!/usr/bin/env python3
"""
Path B Backtester: Trend-following strategy with ML quality filter.

Tests the FULL system on unseen data:
1. H4 trend detection (EMA20 > EMA50 alignment)
2. H1 pullback/breakout entry
3. ML quality filter (take/skip) at configurable threshold
4. ATR-based SL/TP with spread cost

Usage:
    python backtest_path_b.py
"""

import pandas as pd
import numpy as np
import json
import sys
import warnings
from datetime import datetime
from ml.feature_engineering import FeatureEngineering
from ml.ensemble_predictor import EnsemblePredictor

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
SPREAD_COST = 30.0       # $30 spread cost per trade (BTCUSD typical)
LOT_SIZE = 0.01           # 0.01 lots = 0.01 BTC
SL_ATR_MULT = 1.5         # SL = 1.5x ATR
TP_ATR_MULT = 2.0         # TP = 2.0x ATR (R:R = 1.33:1)
MIN_ATR = 50              # Minimum ATR to trade
TEST_DAYS = 90            # Last 90 days for testing


def construct_h4_from_h1(h1_df):
    df = h1_df.copy()
    df['h4_group'] = df['timestamp'].dt.floor('4h')
    h4 = df.groupby('h4_group').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum', 'timestamp': 'first'
    }).reset_index(drop=True)
    return h4


def get_h4_trend(h4_df, current_time):
    available = h4_df[h4_df['timestamp'] < current_time].copy()
    if len(available) < 60:
        return 'neutral'
    available['ema20'] = available['close'].ewm(span=20, adjust=False).mean()
    available['ema50'] = available['close'].ewm(span=50, adjust=False).mean()
    latest = available.iloc[-1]
    price, ema20, ema50 = latest['close'], latest['ema20'], latest['ema50']
    if price > ema20 > ema50:
        return 'bullish'
    elif price < ema20 < ema50:
        return 'bearish'
    return 'neutral'


def check_h1_entry(h1_slice, trend):
    if len(h1_slice) < 30:
        return None, ""
    df = h1_slice.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    latest, prev = df.iloc[-1], df.iloc[-2]
    price, ema20 = latest['close'], latest['ema20']

    if trend == 'bullish':
        if latest['low'] <= ema20 * 1.001 and price > ema20:
            return 'buy', 'pullback'
    elif trend == 'bearish':
        if latest['high'] >= ema20 * 0.999 and price < ema20:
            return 'sell', 'pullback'
    return None, ""


def calc_atr(h1_slice, period=14):
    df = h1_slice.copy()
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0


def simulate_trade(entry_price, direction, atr, h1_future):
    sl_dist = atr * SL_ATR_MULT
    tp_dist = atr * TP_ATR_MULT

    if direction == 'buy':
        sl_price, tp_price = entry_price - sl_dist, entry_price + tp_dist
    else:
        sl_price, tp_price = entry_price + sl_dist, entry_price - tp_dist

    for i, (_, candle) in enumerate(h1_future.iterrows()):
        if direction == 'buy':
            if candle['low'] <= sl_price:
                return 'loss', sl_price, i + 1, sl_dist
            if candle['high'] >= tp_price:
                return 'win', tp_price, i + 1, sl_dist
        else:
            if candle['high'] >= sl_price:
                return 'loss', sl_price, i + 1, sl_dist
            if candle['low'] <= tp_price:
                return 'win', tp_price, i + 1, sl_dist

    if len(h1_future) > 0:
        exit_price = h1_future.iloc[-1]['close']
        pnl = (exit_price - entry_price) if direction == 'buy' else (entry_price - exit_price)
        result = 'win' if pnl > 0 else 'loss'
        return result, exit_price, len(h1_future), sl_dist

    return 'timeout', entry_price, 0, sl_dist


def run_backtest(ml_threshold=None, ml_predictor=None, df_features=None, feature_cols=None,
                 df=None, h4_df=None, test_indices=None):
    """
    Run backtest with optional ML filter at given threshold.
    ml_threshold=None means no ML filter (pure trend).
    """
    label = f"ML@{ml_threshold:.0%}" if ml_threshold is not None else "No Filter"

    trades = []
    skipped_by_ml = 0
    skipped_neutral = 0
    skipped_no_entry = 0
    skipped_low_atr = 0
    cooldown_until_idx = -1

    for idx in test_indices:
        if idx < 50:
            continue
        if idx <= cooldown_until_idx:
            continue

        current_time = df.loc[idx, 'timestamp']

        # 1. H4 trend
        trend = get_h4_trend(h4_df, current_time)
        if trend == 'neutral':
            skipped_neutral += 1
            continue

        # 2. H1 entry signal
        h1_history = df.loc[max(0, idx - 39):idx].copy()
        entry_signal, entry_type = check_h1_entry(h1_history, trend)
        if entry_signal is None:
            skipped_no_entry += 1
            continue

        # 3. ATR check
        atr = calc_atr(h1_history)
        if atr < MIN_ATR:
            skipped_low_atr += 1
            continue

        # 4. ML quality filter
        if ml_threshold is not None and ml_predictor is not None:
            if idx in df_features.index:
                feat_row = df_features.loc[idx]
                features_dict = {f: feat_row[f] for f in feature_cols if f in feat_row.index}
                if any(pd.isna(v) for v in features_dict.values()):
                    skipped_by_ml += 1
                    continue

                # Custom threshold check: avg P(GOOD) across models
                features_scaled = ml_predictor.prepare_features(features_dict)
                good_probs = []
                for name, model in ml_predictor.models.items():
                    probs = model.predict_proba(features_scaled)[0]
                    good_probs.append(probs[1])
                avg_good = np.mean(good_probs)
                good_count = sum(1 for p in good_probs if p > 0.5)

                if good_count < 2 or avg_good < ml_threshold:
                    skipped_by_ml += 1
                    continue
            else:
                skipped_by_ml += 1
                continue

        # 5. Execute trade
        entry_price = df.loc[idx, 'close']
        if entry_signal == 'buy':
            entry_price += SPREAD_COST / 2
        else:
            entry_price -= SPREAD_COST / 2

        future_end = min(idx + 48, len(df) - 1)
        h1_future = df.loc[idx + 1:future_end]
        if len(h1_future) == 0:
            continue

        result, exit_price, bars_held, sl_dist = simulate_trade(entry_price, entry_signal, atr, h1_future)

        # P/L: 0.01 lots of BTCUSD = 0.01 BTC exposure
        # Price move in USD * lot_size = P/L in USD
        if entry_signal == 'buy':
            pnl = (exit_price - entry_price) * LOT_SIZE
        else:
            pnl = (entry_price - exit_price) * LOT_SIZE

        trades.append({
            'time': current_time,
            'direction': entry_signal,
            'entry_type': entry_type,
            'trend': trend,
            'entry': entry_price,
            'exit': exit_price,
            'atr': atr,
            'sl_dist': sl_dist,
            'result': result,
            'pnl': pnl,
            'bars_held': bars_held,
        })

        cooldown_until_idx = idx + bars_held + 3

    # --- Results ---
    if not trades:
        return {'label': label, 'trades': 0}

    trades_df = pd.DataFrame(trades)
    total = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'win'])
    losses = len(trades_df[trades_df['result'] == 'loss'])
    wr = wins / total * 100

    total_pnl = trades_df['pnl'].sum()
    gross_win = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losses > 0 else 0

    cumulative = trades_df['pnl'].cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()
    avg_hold = trades_df['bars_held'].mean()

    buys = trades_df[trades_df['direction'] == 'buy']
    sells = trades_df[trades_df['direction'] == 'sell']
    buy_count = len(buys)
    sell_count = len(sells)
    buy_wr = len(buys[buys['result'] == 'win']) / buy_count * 100 if buy_count > 0 else 0
    sell_wr = len(sells[sells['result'] == 'win']) / sell_count * 100 if sell_count > 0 else 0
    buy_avg_sl = buys['sl_dist'].mean() if buy_count > 0 else 0
    sell_avg_sl = sells['sl_dist'].mean() if sell_count > 0 else 0

    return {
        'label': label,
        'trades': total, 'wins': wins, 'losses': losses, 'wr': wr,
        'pf': pf, 'pnl': total_pnl, 'max_dd': max_dd,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'avg_hold': avg_hold,
        'buy_count': buy_count, 'sell_count': sell_count,
        'buy_wr': buy_wr, 'sell_wr': sell_wr,
        'buy_avg_sl': buy_avg_sl, 'sell_avg_sl': sell_avg_sl,
        'skipped_ml': skipped_by_ml,
        'trades_df': trades_df,
    }


def main():
    print("=" * 70)
    print("  PATH B BACKTEST: Threshold Sweep + BUY/SELL Analysis")
    print("=" * 70)

    # Load data
    df = pd.read_csv('models/training_data_btcusd.csv', parse_dates=['timestamp'])
    print(f"Total H1 candles: {len(df)}")

    test_start = df['timestamp'].max() - pd.Timedelta(days=TEST_DAYS)
    test_df = df[df['timestamp'] >= test_start]
    test_indices = test_df.index.tolist()
    print(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} candles)")

    h4_df = construct_h4_from_h1(df)

    # Load ML
    fe = FeatureEngineering('ml_config.json')
    ml_predictor = EnsemblePredictor('ml_config.json')
    ml_predictor.load_model()
    df_features = fe.add_all_features(df.copy())
    feature_cols = fe.get_feature_columns()

    # --- Check test period trend bias ---
    print(f"\n{'='*70}")
    print("  TEST PERIOD TREND ANALYSIS")
    print(f"{'='*70}")
    test_h4 = construct_h4_from_h1(test_df)
    test_h4['ema20'] = test_h4['close'].ewm(span=20, adjust=False).mean()
    test_h4['ema50'] = test_h4['close'].ewm(span=50, adjust=False).mean()
    bullish_bars = len(test_h4[(test_h4['close'] > test_h4['ema20']) & (test_h4['ema20'] > test_h4['ema50'])])
    bearish_bars = len(test_h4[(test_h4['close'] < test_h4['ema20']) & (test_h4['ema20'] < test_h4['ema50'])])
    neutral_bars = len(test_h4) - bullish_bars - bearish_bars
    total_h4 = len(test_h4)
    print(f"  H4 bars:    {total_h4}")
    print(f"  Bullish:    {bullish_bars} ({bullish_bars/total_h4*100:.0f}%)")
    print(f"  Bearish:    {bearish_bars} ({bearish_bars/total_h4*100:.0f}%)")
    print(f"  Neutral:    {neutral_bars} ({neutral_bars/total_h4*100:.0f}%)")
    price_start = test_df.iloc[0]['close']
    price_end = test_df.iloc[-1]['close']
    pct_change = (price_end - price_start) / price_start * 100
    print(f"  BTC price:  ${price_start:.0f} -> ${price_end:.0f} ({pct_change:+.1f}%)")

    # --- Run backtests at different thresholds ---
    thresholds = [None, 0.50, 0.55, 0.65]
    results = []

    for thresh in thresholds:
        r = run_backtest(
            ml_threshold=thresh, ml_predictor=ml_predictor,
            df_features=df_features, feature_cols=feature_cols,
            df=df, h4_df=h4_df, test_indices=test_indices
        )
        results.append(r)

    # --- Summary Table ---
    print(f"\n{'='*70}")
    print("  THRESHOLD COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'Trades':>7} {'WR':>7} {'PF':>7} {'P/L':>10} {'MaxDD':>10} {'AvgHold':>8}")
    print(f"  {'-'*62}")
    for r in results:
        if r['trades'] == 0:
            print(f"  {r['label']:<12}       0    N/A     N/A        N/A        N/A      N/A")
            continue
        pnl_str = f"${r['pnl']:.2f}"
        dd_str = f"${r['max_dd']:.2f}"
        print(f"  {r['label']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} {pnl_str:>10} {dd_str:>10} {r['avg_hold']:>7.1f}h")

    # --- BUY vs SELL Analysis ---
    print(f"\n{'='*70}")
    print("  BUY vs SELL ANALYSIS (No Filter baseline)")
    print(f"{'='*70}")
    baseline = results[0]
    if baseline['trades'] > 0:
        print(f"  {'Direction':<10} {'Count':>7} {'WR':>7} {'Avg SL$':>10} {'Avg Win$':>10} {'Avg Loss$':>10}")
        print(f"  {'-'*54}")

        tdf = baseline['trades_df']
        for direction in ['buy', 'sell']:
            d = tdf[tdf['direction'] == direction]
            if len(d) == 0:
                continue
            cnt = len(d)
            wr = len(d[d['result'] == 'win']) / cnt * 100
            avg_sl = d['sl_dist'].mean()
            avg_w = d[d['pnl'] > 0]['pnl'].mean() if len(d[d['pnl'] > 0]) > 0 else 0
            avg_l = abs(d[d['pnl'] < 0]['pnl'].mean()) if len(d[d['pnl'] < 0]) > 0 else 0
            print(f"  {direction.upper():<10} {cnt:>7} {wr:>6.1f}% {avg_sl:>10.0f} {avg_w:>10.2f} {avg_l:>10.2f}")

        # Entry type breakdown
        print(f"\n  Entry type breakdown:")
        for etype in ['pullback', 'breakout', 'breakdown']:
            e = tdf[tdf['entry_type'] == etype]
            if len(e) == 0:
                continue
            cnt = len(e)
            wr = len(e[e['result'] == 'win']) / cnt * 100
            print(f"    {etype:<12} {cnt:>4} trades, {wr:.1f}% WR")

    # --- Lot size verification ---
    print(f"\n{'='*70}")
    print("  LOT SIZE VERIFICATION")
    print(f"{'='*70}")
    print(f"  Lot size:       {LOT_SIZE} lots = {LOT_SIZE} BTC")
    print(f"  Example trade:  BTC $100,000 entry, $101,000 exit (+$1,000 move)")
    print(f"  P/L calc:       $1,000 * {LOT_SIZE} = ${1000 * LOT_SIZE:.2f}")
    if baseline['trades'] > 0:
        tdf = baseline['trades_df']
        sample = tdf.iloc[0]
        move = abs(sample['exit'] - sample['entry'])
        print(f"  First trade:    entry=${sample['entry']:.0f}, exit=${sample['exit']:.0f}, move=${move:.0f}, pnl=${sample['pnl']:.2f}")
        print(f"  Verify:         ${move:.0f} * {LOT_SIZE} = ${move * LOT_SIZE:.2f} (matches: {abs(sample['pnl'] - move * LOT_SIZE) < 0.01})")

    print(f"\n{'='*70}")
    print(f"  Backtest complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
