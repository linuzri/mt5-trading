"""
Model Training Module for ML Trading Strategy

This module handles:
- Training Random Forest or XGBoost classifiers
- Model validation and performance metrics
- Hyperparameter tuning
- Saving/loading trained models
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed. Run: pip install xgboost")


class ModelTrainer:
    """Train and evaluate ML trading models"""

    def __init__(self, config_path="ml_config.json"):
        """Initialize with ML configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.model_type = self.config['model_type']
        self.model_params = self.config['model_params']
        self.test_size = self.config['training']['test_size']
        self.val_size = self.config['training']['validation_size']
        self.cv_folds = self.config['training']['cross_validation_folds']

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.performance_metrics = {}

    def prepare_train_test_split(self, df, feature_cols, target_col):
        """
        Split data into train, validation, and test sets using CHRONOLOGICAL order.

        IMPORTANT: Time-series data must NOT be shuffled randomly. Future data must
        never appear in the training set. We use chronological splits:
          - Train: oldest 70% of data
          - Validation: next 15%
          - Test: newest 15%

        Args:
            df: DataFrame with features and labels (must be sorted by time)
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("[i] Splitting data into train/val/test sets (chronological)...")

        # Extract features and labels — data must already be sorted by time
        X = df[feature_cols].values
        y = df[target_col].values

        n = len(X)
        train_end = int(n * (1 - self.test_size - self.val_size))
        val_end = int(n * (1 - self.test_size))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(f"   Train set:      {len(X_train)} samples ({len(X_train)/n*100:.1f}%) [oldest]")
        print(f"   Validation set: {len(X_val)} samples ({len(X_val)/n*100:.1f}%) [middle]")
        print(f"   Test set:       {len(X_test)} samples ({len(X_test)/n*100:.1f}%) [newest]")

        # Store feature names
        self.feature_names = feature_cols

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train, X_val, X_test):
        """
        Standardize features using StandardScaler

        Args:
            X_train, X_val, X_test: Feature arrays

        Returns:
            Scaled X_train, X_val, X_test
        """
        print("[i] Scaling features...")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print("   Fitted StandardScaler on training data")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest classifier

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        print(f"[i] Training Random Forest with {self.model_params['n_estimators']} trees...")
        print("[i] Using class_weight='balanced' to handle class imbalance")

        # Filter params to only those accepted by RandomForestClassifier
        rf_valid_params = {'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                          'max_features', 'random_state', 'n_jobs', 'min_weight_fraction_leaf',
                          'max_leaf_nodes', 'bootstrap', 'oob_score', 'warm_start', 'ccp_alpha'}
        rf_params = {k: v for k, v in self.model_params.items() if k in rf_valid_params}
        self.model = RandomForestClassifier(**rf_params, class_weight='balanced')
        self.model.fit(X_train, y_train)

        print("[OK] Model training complete")

        return self.model

    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost classifier

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        # XGBoost parameters optimized for trading
        xgb_params = {
            'n_estimators': self.model_params.get('n_estimators', 100),
            'max_depth': self.model_params.get('max_depth', 6),
            'learning_rate': self.model_params.get('learning_rate', 0.1),
            'min_child_weight': self.model_params.get('min_child_weight', 3),
            'subsample': self.model_params.get('subsample', 0.8),
            'colsample_bytree': self.model_params.get('colsample_bytree', 0.8),
            'gamma': self.model_params.get('gamma', 0.1),
            'reg_alpha': self.model_params.get('reg_alpha', 0.1),
            'reg_lambda': self.model_params.get('reg_lambda', 1.0),
            'random_state': self.model_params.get('random_state', 42),
            'n_jobs': -1,
            'verbosity': 0
        }

        print(f"[i] Training XGBoost with {xgb_params['n_estimators']} estimators...")
        print(f"   Learning rate: {xgb_params['learning_rate']}, Max depth: {xgb_params['max_depth']}")

        # Equal class weights — no directional bias. Let data speak.
        # Class mapping: 0=SELL, 1=BUY, 2=HOLD
        print("   Class weights: equal (no directional bias)")

        self.model = xgb.XGBClassifier(**xgb_params)
        self.model.fit(X_train, y_train)

        print("[OK] XGBoost training complete")

        return self.model

    def train_model(self, X_train, y_train):
        """
        Train model based on configured model_type

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        if self.model_type == 'xgboost':
            return self.train_xgboost(X_train, y_train)
        else:
            return self.train_random_forest(X_train, y_train)

    def evaluate_model(self, X, y, dataset_name=""):
        """
        Evaluate model performance

        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset (e.g., "Validation", "Test")

        Returns:
            Dictionary of metrics
        """
        print(f"\n[i] Evaluating on {dataset_name} set...")

        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)

        # Handle multi-class metrics
        avg_method = 'weighted'  # Weighted average for multi-class
        precision = precision_score(y, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y, y_pred, average=avg_method, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Print metrics
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        # Print classification report
        print(f"\n   Classification Report:")
        target_names = ['SELL', 'BUY', 'HOLD']
        print(classification_report(y, y_pred, target_names=target_names, zero_division=0))

        # Print confusion matrix
        print(f"   Confusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)

        return metrics

    def cross_validate(self, X_train, y_train):
        """
        Perform time-series cross-validation (walk-forward).

        Uses TimeSeriesSplit instead of random k-fold to respect temporal ordering.
        Each fold trains on past data and validates on the next chronological segment,
        preventing future data leakage.

        Args:
            X_train: Training features (chronologically ordered)
            y_train: Training labels

        Returns:
            CV scores
        """
        print(f"\n[i] Performing {self.cv_folds}-fold TIME-SERIES cross-validation (walk-forward)...")

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=tscv,
            scoring='accuracy'
        )

        print(f"   CV Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"   [NOTE] Walk-forward CV prevents future data leakage in time-series")

        return cv_scores

    def get_feature_importance(self):
        """
        Get feature importance from Random Forest

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n[i] Feature Importance:")
        for idx, row in feature_importance_df.iterrows():
            print(f"   {row['feature']:20s}: {row['importance']:.4f}")

        return feature_importance_df

    def save_model(self):
        """Save trained model and scaler to disk"""
        model_path = self.config['paths']['model_file']
        scaler_path = self.config['paths']['scaler_file']

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        joblib.dump(self.model, model_path)
        print(f"[OK] Saved model to {model_path}")

        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"[OK] Saved scaler to {scaler_path}")

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'performance_metrics': self.performance_metrics,
            'trained_at': datetime.now().isoformat()
        }

        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Saved metadata to {metadata_path}")

    def load_model(self):
        """Load trained model and scaler from disk"""
        model_path = self.config['paths']['model_file']
        scaler_path = self.config['paths']['scaler_file']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                self.performance_metrics = metadata.get('performance_metrics', {})

        print(f"[i] Loaded model from {model_path}")
        return self.model, self.scaler

    def train_and_evaluate(self, df, feature_cols, target_col):
        """
        Complete training pipeline

        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Trained model and performance metrics
        """
        print("=" * 60)
        print("[i] Starting ML Model Training Pipeline")
        print("=" * 60)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_test_split(
            df, feature_cols, target_col
        )

        # Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        # Train model (uses configured model_type: random_forest or xgboost)
        self.train_model(X_train, y_train)

        # Cross-validation
        cv_scores = self.cross_validate(X_train, y_train)

        # Evaluate on validation set
        val_metrics = self.evaluate_model(X_val, y_val, "Validation")

        # Evaluate on test set
        test_metrics = self.evaluate_model(X_test, y_test, "Test")

        # Feature importance
        feature_importance = self.get_feature_importance()

        # Store performance metrics
        self.performance_metrics = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance.to_dict('records')
        }

        # Save model
        self.save_model()

        print("\n" + "=" * 60)
        print("[OK] Training Pipeline Complete!")
        print("=" * 60)

        return self.model, self.performance_metrics


if __name__ == "__main__":
    # Test model training
    print("=" * 60)
    print("ML Model Training Test")
    print("=" * 60)

    # This would be run from train_ml_model.py
    # Just a placeholder for testing
    print("Run train_ml_model.py to train the model")
