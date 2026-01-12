#!/usr/bin/env python3
"""
ML Model Training Script

This script:
1. Loads historical data from MT5
2. Engineers features (technical indicators)
3. Creates labels based on future returns
4. Trains Random Forest classifier
5. Evaluates model performance
6. Saves trained model for live trading

Usage:
    python train_ml_model.py                    # Use cached data
    python train_ml_model.py --refresh          # Download fresh data
    python train_ml_model.py --help             # Show help
"""

import argparse
import json
import sys
from datetime import datetime

# Import ML modules
from ml.data_preparation import DataPreparation
from ml.feature_engineering import FeatureEngineering
from ml.model_trainer import ModelTrainer


def main():
    """Main training pipeline"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ML trading model')
    parser.add_argument('--refresh', action='store_true',
                        help='Force refresh data from MT5 (instead of using cache)')
    parser.add_argument('--config', type=str, default='ml_config.json',
                        help='Path to ML configuration file')
    parser.add_argument('--auth', type=str, default='mt5_auth.json',
                        help='Path to MT5 authentication file')
    args = parser.parse_args()

    print("=" * 70)
    print(" " * 20 + "ML TRADING MODEL TRAINING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print(f"Force refresh: {args.refresh}")
    print("=" * 70)

    try:
        # Load authentication credentials
        print("\n📋 Loading credentials...")
        with open(args.auth, 'r') as f:
            auth = json.load(f)
        print(f"   Login: {auth['login']}")
        print(f"   Server: {auth['server']}")

        # Step 1: Data Preparation
        print("\n" + "=" * 70)
        print("STEP 1: DATA PREPARATION")
        print("=" * 70)

        data_prep = DataPreparation(args.config)
        df = data_prep.get_prepared_data(
            login=auth['login'],
            password=auth['password'],
            server=auth['server'],
            force_refresh=args.refresh
        )

        print(f"\n✅ Data preparation complete")
        print(f"   Total samples: {len(df)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Step 2: Feature Engineering
        print("\n" + "=" * 70)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 70)

        feature_eng = FeatureEngineering(args.config)
        df_features = feature_eng.prepare_features_and_labels(df)

        feature_cols = feature_eng.get_feature_columns()
        target_col = feature_eng.get_target_column()

        print(f"\n✅ Feature engineering complete")
        print(f"   Features: {feature_cols}")
        print(f"   Target: {target_col}")
        print(f"   Final dataset: {len(df_features)} samples")

        # Step 3: Model Training
        print("\n" + "=" * 70)
        print("STEP 3: MODEL TRAINING & EVALUATION")
        print("=" * 70)

        trainer = ModelTrainer(args.config)
        model, metrics = trainer.train_and_evaluate(
            df_features,
            feature_cols,
            target_col
        )

        # Summary
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)

        print("\n📊 Performance Summary:")
        print(f"   Cross-Validation Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        print(f"   Validation Set Accuracy:   {metrics['validation_metrics']['accuracy']:.4f}")
        print(f"   Test Set Accuracy:         {metrics['test_metrics']['accuracy']:.4f}")

        print("\n🎯 Next Steps:")
        print("   1. Review model performance metrics above")
        print("   2. Check feature importance to understand what drives predictions")
        print("   3. If performance is good (>55% accuracy), integrate into trading.py")
        print("   4. Set strategy='ml_random_forest' in config.json to use ML strategy")
        print("   5. Backtest before going live!")

        print("\n💡 Tips:")
        print("   - Re-train weekly: python train_ml_model.py --refresh")
        print("   - Monitor live performance and retrain if accuracy drops")
        print("   - Consider increasing confidence_threshold if getting too many false signals")

        print("\n" + "=" * 70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure mt5_auth.json and ml_config.json exist")
        return 1

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
