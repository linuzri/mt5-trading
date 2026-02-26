"""
Feature Engineering Module for ML Trading Strategy

This module calculates technical indicators and features:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- Bollinger Bands
- Volume indicators
- Price momentum features
- Trend indicators (EMA crossovers, price vs MA)
- Market regime detection (trending vs ranging)
"""

import pandas as pd
import numpy as np
import json


class FeatureEngineering:
    """Calculate technical indicators and features for ML model"""

    def __init__(self, config_path="ml_config.json"):
        """Initialize with ML configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.feature_names = self.config['features']
        self.lookahead_candles = self.config['labeling']['lookahead_candles']
        self.profit_threshold = self.config['labeling']['profit_threshold']
        self.labeling_method = self.config['labeling'].get('method', 'future_return')
        self.stop_loss_threshold = self.config['labeling'].get('stop_loss_threshold', 0.0017)

    def calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()

        return macd_line, macd_signal

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        bb_upper = sma + (std * std_dev)
        bb_lower = sma - (std * std_dev)
        bb_width = (bb_upper - bb_lower) / sma  # Normalized width

        return bb_upper, bb_lower, bb_width

    def calculate_volume_ratio(self, df, period=20):
        """Calculate volume ratio (current volume / average volume)"""
        avg_volume = df['volume'].rolling(window=period).mean()
        volume_ratio = df['volume'] / avg_volume

        return volume_ratio

    def calculate_price_changes(self, df):
        """Calculate price changes over different periods"""
        price_change_1min = df['close'].pct_change(1)  # 1 candle ago
        price_change_5min = df['close'].pct_change(5)  # 5 candles ago

        return price_change_1min, price_change_5min

    def calculate_candlestick_patterns(self, df):
        """
        Calculate candlestick pattern features from OHLC data.

        Returns:
            candle_body_ratio: Body size relative to total range (0-1)
            upper_shadow_ratio: Upper shadow relative to total range (0-1)
            lower_shadow_ratio: Lower shadow relative to total range (0-1)
            engulfing: 1=bullish engulfing, -1=bearish engulfing, 0=none
        """
        candle_range = df['high'] - df['low']
        # Avoid division by zero for doji-like candles with zero range
        safe_range = candle_range.replace(0, np.nan)

        body = (df['close'] - df['open']).abs()
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

        candle_body_ratio = (body / safe_range).fillna(0)
        upper_shadow_ratio = (upper_shadow / safe_range).fillna(0)
        lower_shadow_ratio = (lower_shadow / safe_range).fillna(0)

        # Engulfing pattern: current body fully contains previous body
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        curr_body_high = df[['open', 'close']].max(axis=1)
        curr_body_low = df[['open', 'close']].min(axis=1)
        prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
        prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)

        bullish_engulf = (df['close'] > df['open']) & (prev_close < prev_open) & \
                         (curr_body_low <= prev_body_low) & (curr_body_high >= prev_body_high)
        bearish_engulf = (df['close'] < df['open']) & (prev_close > prev_open) & \
                         (curr_body_low <= prev_body_low) & (curr_body_high >= prev_body_high)

        engulfing = pd.Series(0, index=df.index)
        engulfing[bullish_engulf] = 1
        engulfing[bearish_engulf] = -1

        return candle_body_ratio, upper_shadow_ratio, lower_shadow_ratio, engulfing

    def calculate_trend_features(self, df):
        """
        Calculate trend-related features to detect market direction.
        
        Returns:
            ema_20: 20-period EMA
            ema_50: 50-period EMA
            ema_cross: 1 if EMA20 > EMA50 (bullish), -1 if below (bearish)
            price_vs_ema20: Price position relative to EMA20 (%)
            price_vs_ema50: Price position relative to EMA50 (%)
            trend_strength: ADX-like trend strength (0-100)
        """
        # EMAs
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        
        # EMA crossover signal
        ema_cross = pd.Series(0, index=df.index)
        ema_cross[ema_20 > ema_50] = 1   # Bullish
        ema_cross[ema_20 < ema_50] = -1  # Bearish
        
        # Price position relative to EMAs (normalized)
        price_vs_ema20 = (df['close'] - ema_20) / ema_20
        price_vs_ema50 = (df['close'] - ema_50) / ema_50
        
        # Trend strength (simplified ADX-like calculation)
        # Uses directional movement over rolling window
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self.calculate_atr(df, period=14) * 14  # True range sum
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr.replace(0, np.nan)).fillna(0)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr.replace(0, np.nan)).fillna(0)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        trend_strength = dx.rolling(14).mean().fillna(0)
        
        return ema_cross, price_vs_ema20, price_vs_ema50, trend_strength

    def calculate_regime_features(self, df):
        """
        Detect market regime (trending vs ranging).
        
        Returns:
            regime: 1=trending, 0=ranging (based on BB width and ATR)
            momentum_10: 10-period momentum (rate of change)
            higher_high: 1 if current high > previous 5 highs
            lower_low: 1 if current low < previous 5 lows
        """
        # Momentum (Rate of Change)
        momentum_10 = df['close'].pct_change(10)
        
        # Higher highs / Lower lows (trend structure)
        rolling_high = df['high'].rolling(5).max().shift(1)
        rolling_low = df['low'].rolling(5).min().shift(1)
        
        higher_high = (df['high'] > rolling_high).astype(int)
        lower_low = (df['low'] < rolling_low).astype(int)
        
        # Regime detection: trending if BB is expanding and price moving
        bb_upper, bb_lower, bb_width = self.calculate_bollinger_bands(df)
        bb_width_ma = bb_width.rolling(20).mean()
        
        # Trending = BB width expanding + strong momentum
        is_trending = ((bb_width > bb_width_ma) & (abs(momentum_10) > 0.005)).astype(int)
        
        return is_trending, momentum_10, higher_high, lower_low

    def calculate_multi_timeframe_bias(self, df):
        """
        Calculate higher timeframe trend bias using longer EMAs.
        
        Returns:
            htf_bias: Overall higher timeframe bias (-1 to 1)
            ema_200: 200-period EMA
            price_vs_ema200: Price position relative to EMA200
        """
        # EMA 200 for longer-term trend
        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        price_vs_ema200 = (df['close'] - ema_200) / ema_200
        
        # Higher timeframe bias combines multiple signals
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        ema_100 = df['close'].ewm(span=100, adjust=False).mean()
        
        # Score: +1 for each bullish condition, -1 for bearish
        htf_bias = pd.Series(0.0, index=df.index)
        htf_bias += (df['close'] > ema_50).astype(float) * 0.25
        htf_bias += (df['close'] > ema_100).astype(float) * 0.25
        htf_bias += (df['close'] > ema_200).astype(float) * 0.25
        htf_bias += (ema_50 > ema_200).astype(float) * 0.25
        htf_bias -= (df['close'] < ema_50).astype(float) * 0.25
        htf_bias -= (df['close'] < ema_100).astype(float) * 0.25
        htf_bias -= (df['close'] < ema_200).astype(float) * 0.25
        htf_bias -= (ema_50 < ema_200).astype(float) * 0.25
        
        return htf_bias, price_vs_ema200

    def calculate_crash_features(self, df):
        """
        Calculate features specifically designed to detect crashes/strong moves.
        
        Returns:
            consecutive_direction: Count of consecutive same-direction candles (+ for green, - for red)
            price_velocity: Rate of price change over 5 candles (negative = dropping fast)
            range_position: Where price is in 24h range (0=at low, 1=at high)
            drawdown_pct: Current drawdown from recent high (negative = in drawdown)
        """
        # Consecutive candles in same direction
        candle_direction = (df['close'] > df['open']).astype(int) * 2 - 1  # 1=green, -1=red
        
        # Count consecutive same-direction candles
        consecutive = pd.Series(0, index=df.index, dtype=float)
        count = 0
        prev_dir = 0
        for i in range(len(df)):
            curr_dir = candle_direction.iloc[i]
            if curr_dir == prev_dir:
                count += curr_dir  # Accumulate in direction
            else:
                count = curr_dir  # Reset
            consecutive.iloc[i] = count
            prev_dir = curr_dir
        
        # Price velocity (5-candle momentum, larger negative = crash)
        price_velocity = df['close'].pct_change(5)
        
        # Position within 24h range (288 candles for M5)
        rolling_high = df['high'].rolling(288, min_periods=50).max()
        rolling_low = df['low'].rolling(288, min_periods=50).min()
        range_size = rolling_high - rolling_low
        range_position = (df['close'] - rolling_low) / range_size.replace(0, np.nan)
        range_position = range_position.fillna(0.5)  # Default to middle if no range
        
        # Drawdown from recent high (negative when in drawdown)
        rolling_peak = df['high'].rolling(50).max()
        drawdown_pct = (df['close'] - rolling_peak) / rolling_peak
        
        return consecutive, price_velocity, range_position, drawdown_pct

    def add_all_features(self, df):
        """
        Add all technical indicators and features to DataFrame

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added features
        """
        print("[i] Engineering features...")

        # Make a copy to avoid modifying original
        df = df.copy()

        # RSI
        df['rsi_14'] = self.calculate_rsi(df, period=14)

        # MACD
        df['macd_line'], df['macd_signal'] = self.calculate_macd(df)

        # ATR
        df['atr_14'] = self.calculate_atr(df, period=14)

        # Bollinger Bands
        df['bb_upper'], df['bb_lower'], df['bb_width'] = self.calculate_bollinger_bands(df)

        # Volume ratio
        df['volume_ratio'] = self.calculate_volume_ratio(df)

        # Price changes
        df['price_change_1min'], df['price_change_5min'] = self.calculate_price_changes(df)

        # Candlestick patterns
        df['candle_body_ratio'], df['upper_shadow_ratio'], df['lower_shadow_ratio'], df['engulfing'] = \
            self.calculate_candlestick_patterns(df)

        # Trend features (NEW)
        df['ema_cross'], df['price_vs_ema20'], df['price_vs_ema50'], df['trend_strength'] = \
            self.calculate_trend_features(df)

        # Market regime features (NEW)
        df['is_trending'], df['momentum_10'], df['higher_high'], df['lower_low'] = \
            self.calculate_regime_features(df)

        # Higher timeframe bias (NEW)
        df['htf_bias'], df['price_vs_ema200'] = \
            self.calculate_multi_timeframe_bias(df)

        # Crash detection features (NEW - Feb 6)
        df['consecutive_direction'], df['price_velocity'], df['range_position'], df['drawdown_pct'] = \
            self.calculate_crash_features(df)

        # H1-specific features
        df['hourly_return'] = df['close'].pct_change().fillna(0)

        # Daily range position: where price sits in the ROLLING 24-candle range (H1 = 24 hours)
        # Uses ONLY past data — no look-ahead bias
        rolling_window = 24  # 24 H1 candles = 1 day
        rolling_high = df['high'].rolling(window=rolling_window, min_periods=6).max()
        rolling_low = df['low'].rolling(window=rolling_window, min_periods=6).min()
        rolling_range = rolling_high - rolling_low
        df['daily_range_position'] = ((df['close'] - rolling_low) / rolling_range.replace(0, 1)).fillna(0.5)

        print(f"[OK] Added {len(self.feature_names)} features")

        return df

    def create_labels_sltp_aware(self, df):
        """
        Create trading labels based on SL/TP hits - FIXED BALANCED LOGIC

        Label logic (checks BOTH directions independently):
        - BUY (1): Long TP hit before Long SL (profitable long)
        - SELL (0): Short TP hit before Short SL (profitable short)
        - HOLD (2): Neither direction is clearly profitable

        When BOTH directions would be profitable, choose the one that hits TP FIRST.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with 'label' column
        """
        print(f"[i] Creating BALANCED SL/TP-aware labels (TP={self.profit_threshold:.2%}, SL={self.stop_loss_threshold:.2%})...")

        df = df.copy()
        labels = []

        for i in range(len(df)):
            if i + self.lookahead_candles >= len(df):
                labels.append(2)  # HOLD for rows without enough future data
                continue

            entry_price = df['close'].iloc[i]
            future_data = df.iloc[i+1:i+self.lookahead_candles+1]
            future_highs = future_data['high']
            future_lows = future_data['low']

            # Account for spread cost (~$30 on BTCUSD, or 0.03% at $100K)
            spread_cost_pct = 0.0003  # 0.03% — adjust based on broker

            # Calculate TP and SL price levels (TP must overcome spread)
            tp_price_long = entry_price * (1 + self.profit_threshold + spread_cost_pct)
            sl_price_long = entry_price * (1 - self.stop_loss_threshold)
            tp_price_short = entry_price * (1 - self.profit_threshold - spread_cost_pct)
            sl_price_short = entry_price * (1 + self.stop_loss_threshold)

            # --- Check LONG trade (BUY) ---
            # Find first candle where TP or SL is hit
            long_tp_candle = None
            long_sl_candle = None
            for j, (_, row) in enumerate(future_data.iterrows()):
                if long_tp_candle is None and row['high'] >= tp_price_long:
                    long_tp_candle = j
                if long_sl_candle is None and row['low'] <= sl_price_long:
                    long_sl_candle = j
            
            # Long is profitable if TP hit first (or TP hit and SL never hit)
            long_profitable = False
            if long_tp_candle is not None:
                if long_sl_candle is None or long_tp_candle <= long_sl_candle:
                    long_profitable = True

            # --- Check SHORT trade (SELL) ---
            short_tp_candle = None
            short_sl_candle = None
            for j, (_, row) in enumerate(future_data.iterrows()):
                if short_tp_candle is None and row['low'] <= tp_price_short:
                    short_tp_candle = j
                if short_sl_candle is None and row['high'] >= sl_price_short:
                    short_sl_candle = j
            
            # Short is profitable if TP hit first (or TP hit and SL never hit)
            short_profitable = False
            if short_tp_candle is not None:
                if short_sl_candle is None or short_tp_candle <= short_sl_candle:
                    short_profitable = True

            # --- Determine final label ---
            if long_profitable and short_profitable:
                # Both profitable - choose the one that hits TP first
                if long_tp_candle <= short_tp_candle:
                    labels.append(1)  # BUY (long TP hit first)
                else:
                    labels.append(0)  # SELL (short TP hit first)
            elif long_profitable:
                labels.append(1)  # BUY
            elif short_profitable:
                labels.append(0)  # SELL
            else:
                labels.append(2)  # HOLD

        df['label'] = labels

        # Count label distribution before dropping HOLD
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)
        label_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
        print(f"   Label distribution (before HOLD drop):")
        for label, count in label_counts.items():
            label_name = label_map.get(label, f'Unknown({label})')
            percentage = (count / total) * 100
            print(f"   - {label_name}: {count} ({percentage:.1f}%)")

        # Binary classification: drop HOLD rows (untradeable sideways candles)
        hold_count = len(df[df['label'] == 2])
        df = df[df['label'] != 2].reset_index(drop=True)
        print(f"   Dropped {hold_count} HOLD rows ({hold_count/total*100:.1f}% untradeable)")
        print(f"   Remaining: {len(df)} tradeable samples (BUY + SELL only)")

        # Final binary distribution
        final_counts = df['label'].value_counts().sort_index()
        final_total = len(df)
        print(f"   Final binary distribution:")
        for label, count in final_counts.items():
            label_name = label_map.get(label, f'Unknown({label})')
            percentage = (count / final_total) * 100
            print(f"   - {label_name}: {count} ({percentage:.1f}%)")

        return df

    def create_labels(self, df):
        """
        Create trading labels based on configured method

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with 'label' column
        """
        if self.labeling_method == 'sltp_aware':
            return self.create_labels_sltp_aware(df)
        else:
            # Original future_return method (deprecated for live trading)
            print("[i] Creating labels (future_return method)...")
            print("[!] WARNING: This method doesn't account for SL/TP and may not reflect real profitability!")

            df = df.copy()

            # Calculate future returns (lookahead)
            future_price = df['close'].shift(-self.lookahead_candles)
            future_return = (future_price - df['close']) / df['close']

            # Create labels
            conditions = [
                future_return > self.profit_threshold,   # Strong upward move -> BUY
                future_return < -self.profit_threshold,  # Strong downward move -> SELL
            ]
            choices = [1, 0]  # 1=BUY, 0=SELL

            df['label'] = np.select(conditions, choices, default=2)  # 2=HOLD

            # Count label distribution
            label_counts = df['label'].value_counts().sort_index()
            print(f"   Label distribution:")
            label_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
            for label, count in label_counts.items():
                label_name = label_map.get(label, f'Unknown({label})')
                percentage = (count / len(df)) * 100
                print(f"   - {label_name}: {count} ({percentage:.1f}%)")

            return df

    def prepare_features_and_labels(self, df):
        """
        Prepare final dataset with features and labels

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            DataFrame ready for ML training
        """
        # Add features
        df = self.add_all_features(df)

        # Create labels
        df = self.create_labels(df)

        # Drop rows with NaN (from rolling windows and future lookahead)
        original_len = len(df)
        df = df.dropna()
        dropped = original_len - len(df)

        print(f"[i] Dropped {dropped} rows with NaN values ({len(df)} remaining)")

        # Verify we have all required features
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        print(f"[OK] Dataset ready: {len(df)} samples with {len(self.feature_names)} features")

        return df

    def get_feature_columns(self):
        """Get list of feature column names"""
        return self.feature_names

    def get_target_column(self):
        """Get target column name"""
        return 'label'


if __name__ == "__main__":
    # Test feature engineering
    print("=" * 60)
    print("ML Feature Engineering Test")
    print("=" * 60)

    # Load sample data
    from data_preparation import DataPreparation

    with open("../mt5_auth.json", "r") as f:
        auth = json.load(f)

    # Get data
    data_prep = DataPreparation("../ml_config.json")
    df = data_prep.get_prepared_data(
        login=auth['login'],
        password=auth['password'],
        server=auth['server']
    )

    # Engineer features
    feature_eng = FeatureEngineering("../ml_config.json")
    df_features = feature_eng.prepare_features_and_labels(df)

    # Display results
    print("\n" + "=" * 60)
    print("Feature Engineering Results:")
    print("=" * 60)
    print(f"\nFeatures: {feature_eng.get_feature_columns()}")
    print(f"\nTarget: {feature_eng.get_target_column()}")
    print(f"\nSample rows:")
    print(df_features[feature_eng.get_feature_columns() + ['label']].head(10))
    print(f"\nFeature statistics:")
    print(df_features[feature_eng.get_feature_columns()].describe())
