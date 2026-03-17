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

    def _balance_training_data(self, X, y):
        """Downsample majority class to match minority class count."""
        classes, counts = np.unique(y, return_counts=True)
        min_count = min(counts)
        balanced_indices = []
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            sampled = np.random.choice(cls_indices, size=min_count, replace=False)
            balanced_indices.extend(sampled)
        np.random.shuffle(balanced_indices)
        print(f"    Balanced: {len(X)} -> {len(balanced_indices)} samples ({min_count} per class)")
        return X[balanced_indices], y[balanced_indices]

    def _train_rf(self, X_train, y_train):
        """Train Random Forest"""
        print("\n[i] Training Random Forest (n_estimators=200, max_depth=6)...")
        X_bal, y_bal = self._balance_training_data(X_train, y_train)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_bal, y_bal)
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

        X_bal, y_bal = self._balance_training_data(X_train, y_train)
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_bal, y_bal)
        print("[OK] XGBoost trained")
        return model

    def _train_lgb(self, X_train, y_train):
        """Train LightGBM"""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        print("\n[i] Training LightGBM (n_estimators=200, max_depth=6, lr=0.05)...")

        X_bal, y_bal = self._balance_training_data(X_train, y_train)
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        # Pass feature names via DataFrame to avoid sklearn warning
        import pandas as _pd
        X_train_df = _pd.DataFrame(X_bal, columns=self.feature_names)
        model.fit(X_train_df, y_bal)
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
        # Detect binary vs 3-class from unique labels
        unique_labels = sorted(set(y_test) | set(ensemble_pred))
        if len(unique_labels) <= 2:
            target_names = ['SELL', 'BUY']
        else:
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

        # Walk-forward validation for realistic performance estimate
        print("\n")
        wf_metrics = self.walk_forward_validate(df, feature_cols, target_col)
        self.performance_metrics.update(wf_metrics)

        # Feature importance analysis
        top_features = self.get_feature_importance(top_n=15)
        self.performance_metrics['top_features'] = top_features

        print("\n" + "=" * 60)
        print("[OK] Ensemble Training Pipeline Complete!")
        print("=" * 60)

        return self.models, self.performance_metrics

    def walk_forward_validate(self, df, feature_cols, target_col, n_splits=5, train_months=6, test_weeks=4):
        """
        Walk-forward validation: train on rolling window, test on next period.
        Gives a realistic estimate of model performance on unseen future data.
        """
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler

        print("=" * 60)
        print(f"[i] Walk-Forward Validation ({n_splits} splits)")
        print(f"    Train window: {train_months} months, Test window: {test_weeks} weeks")
        print("=" * 60)

        X = df[feature_cols].values
        y = df[target_col].values

        # Auto-detect candles per day from data density
        total_days = (len(X) / max(1, len(X))) * 365  # rough estimate
        if len(df) > 0 and 'timestamp' in df.columns:
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            candles_per_day = max(1, len(df) // max(1, date_range))
        else:
            candles_per_day = 24  # Default H1
        print(f"    Auto-detected ~{candles_per_day} candles/day")
        train_size = train_months * 30 * candles_per_day
        test_size = test_weeks * 7 * candles_per_day
        step_size = test_size

        total_needed = train_size + n_splits * step_size
        if len(X) < total_needed:
            n_splits = max(1, (len(X) - train_size) // step_size)
            print(f"[WARN] Reduced to {n_splits} splits (not enough data)")

        split_accuracies = []

        for i in range(n_splits):
            test_end = len(X) - i * step_size
            test_start = test_end - test_size
            train_end = test_start
            train_start = max(0, train_end - train_size)

            if train_start >= train_end or test_start >= test_end:
                break

            X_train_wf = X[train_start:train_end]
            y_train_wf = y[train_start:train_end]
            X_test_wf = X[test_start:test_end]
            y_test_wf = y[test_start:test_end]

            scaler_wf = StandardScaler()
            X_train_s = scaler_wf.fit_transform(X_train_wf)
            X_test_s = scaler_wf.transform(X_test_wf)

            # Balance training data for each split
            rf = self._train_rf(X_train_s, y_train_wf)
            xgb_m = self._train_xgb(X_train_s, y_train_wf)
            lgb_m = self._train_lgb(X_train_s, y_train_wf)

            preds = [rf.predict(X_test_s), xgb_m.predict(X_test_s), lgb_m.predict(X_test_s)]
            ensemble_pred = self._ensemble_predict(preds)

            acc = accuracy_score(y_test_wf, ensemble_pred)
            split_accuracies.append(acc)
            print(f"\n  Split {i+1}: Train[{train_start}:{train_end}] Test[{test_start}:{test_end}] | Accuracy: {acc:.4f}")

        avg_accuracy = np.mean(split_accuracies)
        std_accuracy = np.std(split_accuracies)

        print(f"\n{'=' * 60}")
        print(f"  Walk-Forward Results:")
        print(f"  Average Accuracy: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"  Min: {min(split_accuracies):.4f}, Max: {max(split_accuracies):.4f}")
        print(f"  Break-even threshold (with costs): ~0.48")
        if avg_accuracy < 0.48:
            print(f"  [WARN] Average accuracy below break-even! Model may not be profitable.")
        print(f"{'=' * 60}")

        return {
            'walk_forward_accuracies': split_accuracies,
            'walk_forward_mean': float(avg_accuracy),
            'walk_forward_std': float(std_accuracy),
        }

    def get_feature_importance(self, top_n=15):
        """Get top N most important features across all models."""
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(self.feature_names):
                    if feat not in importances:
                        importances[feat] = 0
                    importances[feat] += model.feature_importances_[i]

        for feat in importances:
            importances[feat] /= len(self.models)

        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        print(f"\n[i] Feature Importance (top {top_n}):")
        for feat, imp in sorted_feats[:top_n]:
            print(f"    {feat:25s} {imp:.4f}")
        print(f"\n[i] Low-importance features (candidates for removal):")
        for feat, imp in sorted_feats[top_n:]:
            print(f"    {feat:25s} {imp:.4f}")

        return [feat for feat, _ in sorted_feats[:top_n]]


if __name__ == "__main__":
    print("Run train_ml_model.py --ensemble to train the ensemble model")
