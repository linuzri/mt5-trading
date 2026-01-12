"""
Feature Engineering Module for ML Trading Strategy

This module calculates technical indicators and features:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- Bollinger Bands
- Volume indicators
- Price momentum features
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

    def add_all_features(self, df):
        """
        Add all technical indicators and features to DataFrame

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added features
        """
        print("🔧 Engineering features...")

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

        print(f"✅ Added {len(self.feature_names)} features")

        return df

    def create_labels(self, df):
        """
        Create trading labels based on future returns

        Label logic:
        - BUY (1): If price increases > profit_threshold in next N candles
        - SELL (0): If price decreases > profit_threshold in next N candles
        - HOLD (2): Otherwise (price moves within threshold)

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with 'label' column
        """
        print("🏷️ Creating labels...")

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

        print(f"🧹 Dropped {dropped} rows with NaN values ({len(df)} remaining)")

        # Verify we have all required features
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        print(f"✅ Dataset ready: {len(df)} samples with {len(self.feature_names)} features")

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
