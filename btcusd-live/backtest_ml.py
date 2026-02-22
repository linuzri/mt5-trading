"""
ML Strategy Backtester for BTCUSD H1
Simulates the ensemble ML strategy with realistic spread costs.

Usage:
    python backtest_ml.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.feature_engineering import FeatureEngineering
from ml.data_preparation import DataPreparation


class MLBacktester:
    def __init__(self, config_path="config.json", ml_config_path="ml_config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        with open(ml_config_path) as f:
            self.ml_config = json.load(f)

        # Load trained models
        self.models = {
            'rf': joblib.load('models/ensemble_rf.pkl'),
            'xgb': joblib.load('models/ensemble_xgb.pkl'),
            'lgb': joblib.load('models/ensemble_lgb.pkl'),
        }
        self.scaler = joblib.load('models/ensemble_scaler.pkl')
        self.feature_names = self.ml_config['features']
        self.label_map = {0: 'sell', 1: 'buy'}

        # Thresholds
        self.confidence_threshold = self.ml_config['prediction']['confidence_threshold']
        self.min_prob_diff = self.ml_config['prediction']['min_probability_diff']
        self.tp_pct = self.ml_config['labeling']['profit_threshold']
        self.sl_pct = self.ml_config['labeling']['stop_loss_threshold']
        self.lookahead = self.ml_config['labeling']['lookahead_candles']

    def predict(self, features_row):
        """Get ensemble prediction for a single row."""
        feat_values = np.array([features_row[f] for f in self.feature_names]).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_values)

        signals = {}
        probs_list = []
        for name, model in self.models.items():
            probs = model.predict_proba(feat_scaled)[0]
            probs_list.append(probs)
            sell_p, buy_p = probs[0], probs[1]
            if buy_p > sell_p:
                signals[name] = ('buy', buy_p)
            else:
                signals[name] = ('sell', sell_p)

        # Majority vote
        from collections import Counter
        votes = [s[0] for s in signals.values()]
        vote_counts = Counter(votes)
        winner, count = vote_counts.most_common(1)[0]

        if count < 2:
            return None, 0.0  # No consensus

        # Average confidence of agreeing models
        agreeing_confs = [signals[n][1] for n in signals if signals[n][0] == winner]
        confidence = np.mean(agreeing_confs)

        if confidence < self.confidence_threshold:
            return None, confidence

        # Check prob diff
        avg_probs = np.mean(probs_list, axis=0)
        prob_diff = abs(avg_probs[1] - avg_probs[0])
        if prob_diff < self.min_prob_diff:
            return None, confidence

        return winner, confidence

    def simulate_trade(self, df, entry_idx, signal, lot_size=0.01, spread_cost=0.30):
        """Simulate a single trade with SL/TP on future candles.
        
        At 0.01 lots (0.01 BTC):
        - TP=0.5% ($500 on 1 BTC) = $5.00 per trade
        - SL=0.4% ($400 on 1 BTC) = $4.00 per trade
        - Spread: ~$0.30 per trade at 0.01 lots
        """
        entry_price = df['close'].iloc[entry_idx]

        if signal == 'buy':
            tp_price = entry_price * (1 + self.tp_pct)
            sl_price = entry_price * (1 - self.sl_pct)
        else:  # sell
            tp_price = entry_price * (1 - self.tp_pct)
            sl_price = entry_price * (1 + self.sl_pct)

        # Walk forward through candles
        end_idx = min(entry_idx + self.lookahead, len(df))
        for j in range(entry_idx + 1, end_idx):
            high = df['high'].iloc[j]
            low = df['low'].iloc[j]

            if signal == 'buy':
                if high >= tp_price:
                    pnl = (tp_price - entry_price) * lot_size - spread_cost
                    return {'signal': signal, 'entry': entry_price, 'exit': tp_price,
                            'pnl': pnl, 'result': 'TP', 'bars_held': j - entry_idx}
                if low <= sl_price:
                    pnl = (sl_price - entry_price) * lot_size - spread_cost
                    return {'signal': signal, 'entry': entry_price, 'exit': sl_price,
                            'pnl': pnl, 'result': 'SL', 'bars_held': j - entry_idx}
            else:  # sell
                if low <= tp_price:
                    pnl = (entry_price - tp_price) * lot_size - spread_cost
                    return {'signal': signal, 'entry': entry_price, 'exit': tp_price,
                            'pnl': pnl, 'result': 'TP', 'bars_held': j - entry_idx}
                if high >= sl_price:
                    pnl = (entry_price - sl_price) * lot_size - spread_cost
                    return {'signal': signal, 'entry': entry_price, 'exit': sl_price,
                            'pnl': pnl, 'result': 'SL', 'bars_held': j - entry_idx}

        # Timeout — close at last candle's close
        exit_price = df['close'].iloc[end_idx - 1]
        if signal == 'buy':
            pnl = (exit_price - entry_price) * lot_size - spread_cost
        else:
            pnl = (entry_price - exit_price) * lot_size - spread_cost
        return {'signal': signal, 'entry': entry_price, 'exit': exit_price,
                'pnl': pnl, 'result': 'TIMEOUT', 'bars_held': end_idx - entry_idx}

    def run(self, lot_size=0.01, spread_cost=0.30, initial_balance=200.0):
        """Run full backtest on the last 30% of data (test set).
        
        Args:
            lot_size: 0.01 = 0.01 BTC. At BTC $100K: TP=$5, SL=$4 per trade.
            spread_cost: $0.30 per trade at 0.01 lots (~$30 spread on 1 BTC)
            initial_balance: Starting account balance
        """
        print("=" * 60)
        print("ML Strategy Backtest — BTCUSD H1")
        print(f"Lot size: {lot_size} | Spread: ${spread_cost}/trade")
        print(f"TP: {self.tp_pct:.1%} | SL: {self.sl_pct:.1%}")
        print(f"At BTC ~$100K: TP=${100000*self.tp_pct*lot_size:.2f}, SL=${100000*self.sl_pct*lot_size:.2f} per trade")
        print(f"Confidence: {self.confidence_threshold:.0%} | Min prob diff: {self.min_prob_diff:.0%}")
        print(f"Initial balance: ${initial_balance:.2f}")
        print("=" * 60)

        # Load data and add features
        df = pd.read_csv('models/training_data_btcusd.csv', parse_dates=['timestamp'])
        fe = FeatureEngineering('ml_config.json')
        df = fe.add_all_features(df)

        # Use last 30% as backtest period (same as test set)
        split_idx = int(len(df) * 0.7)
        df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
        print(f"\nBacktest period: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}")
        print(f"Candles: {len(df_test)}")

        trades = []
        equity = [initial_balance]
        cooldown = 0
        max_daily_trades = self.ml_config.get('risk_management', {}).get('max_trades_per_day', 3)

        # Track daily trades
        current_day = None
        daily_count = 0

        for i in range(100, len(df_test) - self.lookahead):
            # Cooldown
            if cooldown > 0:
                cooldown -= 1
                continue

            # Daily trade limit
            day = df_test['timestamp'].iloc[i].date()
            if day != current_day:
                current_day = day
                daily_count = 0
            if daily_count >= max_daily_trades:
                continue

            # Get prediction
            row = df_test.iloc[i]
            signal, confidence = self.predict(row)

            if signal is None:
                continue

            # Simulate trade
            result = self.simulate_trade(df_test, i, signal, lot_size, spread_cost)
            result['timestamp'] = str(df_test['timestamp'].iloc[i])
            result['confidence'] = confidence
            trades.append(result)

            equity.append(equity[-1] + result['pnl'])
            daily_count += 1

            # Cooldown after trade (in candles, not seconds)
            cooldown = 1  # Skip at least 1 candle

        return self.report(trades, equity, initial_balance)

    def report(self, trades, equity, initial_balance):
        """Print backtest results."""
        if not trades:
            print("\n[!] No trades executed during backtest period.")
            return {}

        df_trades = pd.DataFrame(trades)
        total = len(trades)
        wins = len(df_trades[df_trades['pnl'] > 0])
        losses = len(df_trades[df_trades['pnl'] <= 0])
        win_rate = wins / total if total > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].mean()) if losses > 0 else 0
        profit_factor = (df_trades[df_trades['pnl'] > 0]['pnl'].sum() /
                        abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())) if losses > 0 else float('inf')

        # Max drawdown
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown = drawdown.max()

        # Sharpe ratio (daily returns)
        returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Direction breakdown
        buys = df_trades[df_trades['signal'] == 'buy']
        sells = df_trades[df_trades['signal'] == 'sell']
        buy_wr = len(buys[buys['pnl'] > 0]) / len(buys) if len(buys) > 0 else 0
        sell_wr = len(sells[sells['pnl'] > 0]) / len(sells) if len(sells) > 0 else 0

        # Result type breakdown
        tp_count = len(df_trades[df_trades['result'] == 'TP'])
        sl_count = len(df_trades[df_trades['result'] == 'SL'])
        timeout_count = len(df_trades[df_trades['result'] == 'TIMEOUT'])

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total trades:     {total}")
        print(f"  Wins / Losses:    {wins} / {losses}")
        print(f"  Win Rate:         {win_rate:.1%}")
        print(f"  Profit Factor:    {profit_factor:.2f}")
        print(f"  Total P/L:        ${total_pnl:.2f}")
        print(f"  Avg Win:          ${avg_win:.2f}")
        print(f"  Avg Loss:         ${avg_loss:.2f}")
        print(f"  Max Drawdown:     {max_drawdown:.1%}")
        print(f"  Sharpe Ratio:     {sharpe:.2f}")
        print(f"  Final Equity:     ${equity[-1]:.2f} (from ${equity[0]:.2f})")
        print(f"  Return:           {(equity[-1]/equity[0]-1)*100:.1f}%")
        print(f"  Equity Low:       ${min(equity):.2f}")
        print(f"  Max DD ($):       ${max(peak - equity_arr):.2f}")
        print(f"")
        print(f"  BUY trades:       {len(buys)} (WR: {buy_wr:.1%})")
        print(f"  SELL trades:      {len(sells)} (WR: {sell_wr:.1%})")
        print(f"  TP / SL / Timeout: {tp_count} / {sl_count} / {timeout_count}")
        print(f"  Avg bars held:    {df_trades['bars_held'].mean():.1f}")
        print("=" * 60)

        return {
            'total_trades': total,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'final_equity': equity[-1],
        }


if __name__ == "__main__":
    bt = MLBacktester()
    # 0.01 lots = 0.01 BTC. Spread ~$30 on 1 BTC = $0.30 on 0.01 lots.
    results = bt.run(lot_size=0.01, spread_cost=0.30, initial_balance=200.0)
