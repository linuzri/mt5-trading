"""
Data Preparation Module for ML Trading Strategy

This module handles:
- Extracting historical OHLCV data from MT5
- Data cleaning and validation
- Saving/loading prepared datasets
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import json
import os


class DataPreparation:
    """Handles data extraction and preparation from MT5"""

    def __init__(self, config_path="ml_config.json"):
        """Initialize with ML configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.symbol = self.config['symbol']
        self.timeframe_str = self.config['timeframe']
        self.training_days = self.config['data_collection']['training_period_days']
        self.min_data_points = self.config['data_collection']['min_data_points']

        # Map timeframe string to MT5 constant
        self.timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        self.timeframe = self.timeframe_map.get(self.timeframe_str, mt5.TIMEFRAME_M1)

    def connect_mt5(self, login, password, server):
        """Connect to MT5 terminal"""
        if not mt5.initialize(login=login, password=password, server=server):
            error = mt5.last_error()
            raise Exception(f"MT5 initialization failed: {error}")

        if not mt5.symbol_select(self.symbol, True):
            mt5.shutdown()
            raise Exception(f"Failed to select symbol {self.symbol}")

        print(f"[OK] Connected to MT5: {mt5.terminal_info().name}")
        return True

    def extract_historical_data(self, days=None):
        """
        Extract historical OHLCV data from MT5

        Args:
            days: Number of days of historical data (default from config)

        Returns:
            pandas DataFrame with OHLCV data
        """
        if days is None:
            days = self.training_days

        # Calculate start date
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        print(f"[i] Extracting {days} days of {self.symbol} data on {self.timeframe_str}...")
        print(f"   From: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   To:   {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

        # Get rates from MT5
        rates = mt5.copy_rates_range(
            self.symbol,
            self.timeframe,
            start_date,
            end_date
        )

        if rates is None or len(rates) == 0:
            raise Exception(f"Failed to get rates for {self.symbol}: {mt5.last_error()}")

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Rename columns for clarity
        df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        }, inplace=True)

        # Use tick_volume as volume (more reliable on crypto/forex)
        if 'volume' not in df.columns:
            df['volume'] = df['tick_volume']

        print(f"[OK] Extracted {len(df)} candles")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def validate_data(self, df):
        """
        Validate data quality

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if data is valid
        """
        issues = []

        # Check minimum data points
        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data: {len(df)} < {self.min_data_points}")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")

        # Check for duplicate timestamps
        duplicates = df.duplicated(subset=['timestamp']).sum()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps: {duplicates}")

        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"Invalid prices in {col}: found zero or negative values")

        # Check for zero volume
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > len(df) * 0.5:  # More than 50% zero volume is suspicious
            issues.append(f"High zero volume count: {zero_volume}/{len(df)}")

        # Print validation results
        if issues:
            print("[!] Data validation warnings:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("[OK] Data validation passed")
            return True

    def clean_data(self, df):
        """
        Clean and prepare data

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        print("[i] Cleaning data...")

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        if len(df) < original_len:
            print(f"   Removed {original_len - len(df)} duplicate rows")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Forward fill missing values (if any)
        df = df.ffill().bfill()

        # Remove any remaining NaN rows
        df = df.dropna()

        # Ensure volume is non-negative
        df.loc[df['volume'] < 0, 'volume'] = 0

        print(f"[OK] Cleaned data: {len(df)} rows remaining")

        return df

    def save_data(self, df, filepath=None):
        """Save prepared data to CSV"""
        if filepath is None:
            filepath = self.config['paths']['training_data']

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath, index=False)
        print(f"[OK] Saved data to {filepath}")

    def load_data(self, filepath=None):
        """Load prepared data from CSV"""
        if filepath is None:
            filepath = self.config['paths']['training_data']

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"[i] Loaded {len(df)} rows from {filepath}")
        return df

    def get_prepared_data(self, login, password, server, force_refresh=False):
        """
        Get prepared data (load from cache or extract fresh)

        Args:
            login: MT5 login
            password: MT5 password
            server: MT5 server
            force_refresh: Force fresh data extraction

        Returns:
            pandas DataFrame
        """
        filepath = self.config['paths']['training_data']

        # Try to load cached data
        if not force_refresh and os.path.exists(filepath):
            try:
                df = self.load_data(filepath)
                print("[i] Using cached data. Use force_refresh=True to download fresh data.")
                return df
            except Exception as e:
                print(f"[!] Failed to load cached data: {e}")
                print("   Extracting fresh data...")

        # Extract fresh data
        self.connect_mt5(login, password, server)

        try:
            df = self.extract_historical_data()
            self.validate_data(df)
            df = self.clean_data(df)
            self.save_data(df)
            return df
        finally:
            mt5.shutdown()
            print("[i] Disconnected from MT5")


if __name__ == "__main__":
    # Test data preparation
    print("=" * 60)
    print("ML Data Preparation Test")
    print("=" * 60)

    # Load credentials
    with open("mt5_auth.json", "r") as f:
        auth = json.load(f)

    # Initialize data preparation
    data_prep = DataPreparation("ml_config.json")

    # Get prepared data
    df = data_prep.get_prepared_data(
        login=auth['login'],
        password=auth['password'],
        server=auth['server'],
        force_refresh=True
    )

    # Display summary
    print("\n" + "=" * 60)
    print("Data Summary:")
    print("=" * 60)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total candles: {len(df)}")
