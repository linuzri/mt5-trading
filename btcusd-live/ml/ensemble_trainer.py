"""
Ensemble Model Training Module for ML Trading Strategy

Trains 3 models (Random Forest, XGBoost, LightGBM) on the same data
and reports individual + ensemble (majority vote) accuracy.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARN] LightGBM not installed. Run: pip install lightgbm")


class EnsembleTrainer:
    """Train and evaluate ensemble of 3 ML trading models"""

    def __init__(self, config_path="ml_config.json"):
        """Initialize with ML configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.model_params = self.config['model_params']
        self.test_size = self.config['training']['test_size']
        self.val_size = self.config['training']['validation_size']

        # Ensemble paths
        ensemble_paths = self.config.get('ensemble_paths', {})
        self.rf_path = ensemble_paths.get('rf_model', 'models/ensemble_rf.pkl')
        self.xgb_path = ensemble_paths.get('xgb_model', 'models/ensemble_xgb.pkl')
        self.lgb_path = ensemble_paths.get('lgb_model', 'models/ensemble_lgb.pkl')
        self.scaler_path = ensemble_paths.get('scaler', 'models/ensemble_scaler.pkl')
        self.metadata_path = ensemble_paths.get('metadata', 'models/ensemble_metadata.json')

        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.performance_metrics = {}

    def prepare_train_test_split(self, df, feature_cols, target_col):
        """Split data chronologically into train/val/test"""
        print("[i] Splitting data into train/val/test sets (chronological)...")

        X = df[feature_cols].values
        y = df[target_col].values
        n = len(X)

        train_end = int(n * (1 - self.test_size - self.val_size))
        val_end = int(n * (1 - self.test_size))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(f"   Train:      {len(X_train)} samples ({len(X_train)/n*100:.1f}%)")
        print(f"   Validation: {len(X_val)} samples ({len(X_val)/n*100:.1f}%)")
        print(f"   Test:       {len(X_test)} samples ({len(X_test)/n*100:.1f}%)")

        self.feature_names = feature_cols
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train, X_val, X_test):
        """Standardize features"""
        print("[i] Scaling features...")
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s

    def _train_rf(self, X_train, y_train):
        """Train Random Forest"""
        print("\n[i] Training Random Forest (n_estimators=200, max_depth=6)...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        print("[OK] Random Forest trained")
        return model

    def _train_xgb(self, X_train, y_train):
        """Train XGBoost"""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        xgb_params = {
            'n_estimators': self.model_params.get('n_estimators', 200),
            'max_depth': self.model_params.get('max_depth', 6),
            'learning_rate': self.model_params.get('learning_rate', 0.05),
            'min_child_weight': self.model_params.get('min_child_weight', 3),
            'subsample': self.model_params.get('subsample', 0.8),
            'colsample_bytree': self.model_params.get('colsample_bytree', 0.8),
            'gamma': self.model_params.get('gamma', 0.1),
            'reg_alpha': self.model_params.get('reg_alpha', 0.1),
            'reg_lambda': self.model_params.get('reg_lambda', 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        print(f"\n[i] Training XGBoost (n_estimators={xgb_params['n_estimators']}, lr={xgb_params['learning_rate']})...")

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        print("[OK] XGBoost trained")
        return model

    def _train_lgb(self, X_train, y_train):
        """Train LightGBM"""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        print("\n[i] Training LightGBM (n_estimators=200, max_depth=6, lr=0.05)...")

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        # Pass feature names via DataFrame to avoid sklearn warning
        import pandas as _pd
        X_train_df = _pd.DataFrame(X_train, columns=self.feature_names)
        model.fit(X_train_df, y_train)
        print("[OK] LightGBM trained")
        return model

    def _evaluate_model(self, model, X, y, name=""):
        """Evaluate a single model"""
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"   {name} accuracy: {acc:.4f}")
        return acc, y_pred

    def _ensemble_predict(self, model_preds):
        """Majority vote across model predictions"""
        ensemble_preds = []
        for i in range(len(model_preds[0])):
            votes = [preds[i] for preds in model_preds]
            counter = Counter(votes)
            winner, count = counter.most_common(1)[0]
            ensemble_preds.append(winner)
        return np.array(ensemble_preds)

    def save_models(self):
        """Save all models, scaler, and metadata"""
        os.makedirs(os.path.dirname(self.rf_path) or '.', exist_ok=True)

        for name, path in [('rf', self.rf_path), ('xgb', self.xgb_path), ('lgb', self.lgb_path)]:
            joblib.dump(self.models[name], path)
            print(f"[OK] Saved {name.upper()} model to {path}")

        joblib.dump(self.scaler, self.scaler_path)
        print(f"[OK] Saved scaler to {self.scaler_path}")

        metadata = {
            'model_type': 'ensemble',
            'models': ['rf', 'xgb', 'lgb'],
            'feature_names': self.feature_names,
            'individual_accuracies': self.performance_metrics.get('individual_accuracies', {}),
            'ensemble_accuracy': self.performance_metrics.get('ensemble_test_accuracy', 0),
            'performance_metrics': self.performance_metrics,
            'trained_at': datetime.now().isoformat()
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Saved metadata to {self.metadata_path}")

    def train_and_evaluate(self, df, feature_cols, target_col):
        """
        Complete ensemble training pipeline

        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            models dict and performance metrics
        """
        print("=" * 60)
        print("[i] Starting ENSEMBLE Model Training Pipeline")
        print("   Models: Random Forest + XGBoost + LightGBM")
        print("=" * 60)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_test_split(
            df, feature_cols, target_col
        )

        # Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        # Train all 3 models
        self.models['rf'] = self._train_rf(X_train, y_train)
        self.models['xgb'] = self._train_xgb(X_train, y_train)
        self.models['lgb'] = self._train_lgb(X_train, y_train)

        # Evaluate each model on validation set
        print("\n" + "-" * 40)
        print("[i] Validation Set Results:")
        val_accuracies = {}
        for name in ['rf', 'xgb', 'lgb']:
            acc, _ = self._evaluate_model(self.models[name], X_val, y_val, name.upper())
            val_accuracies[name] = acc

        # Evaluate each model on test set
        print("\n" + "-" * 40)
        print("[i] Test Set Results:")
        test_accuracies = {}
        test_preds = []
        for name in ['rf', 'xgb', 'lgb']:
            acc, y_pred = self._evaluate_model(self.models[name], X_test, y_test, name.upper())
            test_accuracies[name] = acc
            test_preds.append(y_pred)

        # Ensemble accuracy (majority vote on test set)
        ensemble_pred = self._ensemble_predict(test_preds)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        print(f"\n   ENSEMBLE (majority vote) accuracy: {ensemble_acc:.4f}")

        # Classification report for ensemble
        print(f"\n   Ensemble Classification Report:")
        target_names = ['SELL', 'BUY', 'HOLD']
        print(classification_report(y_test, ensemble_pred, target_names=target_names, zero_division=0))

        # Store metrics
        self.performance_metrics = {
            'individual_accuracies': {k: float(v) for k, v in test_accuracies.items()},
            'validation_accuracies': {k: float(v) for k, v in val_accuracies.items()},
            'ensemble_test_accuracy': float(ensemble_acc),
            'test_metrics': {'accuracy': float(ensemble_acc)},
            'cv_mean': float(np.mean(list(test_accuracies.values()))),
            'cv_std': float(np.std(list(test_accuracies.values()))),
        }

        # Save all models
        self.save_models()

        print("\n" + "=" * 60)
        print("[OK] Ensemble Training Pipeline Complete!")
        print("=" * 60)

        return self.models, self.performance_metrics


if __name__ == "__main__":
    print("Run train_ml_model.py --ensemble to train the ensemble model")
