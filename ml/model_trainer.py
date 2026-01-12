"""
Model Training Module for ML Trading Strategy

This module handles:
- Training Random Forest classifier
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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
        Split data into train, validation, and test sets

        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("📊 Splitting data into train/val/test sets...")

        # Extract features and labels
        X = df[feature_cols].values
        y = df[target_col].values

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=42,
            stratify=y  # Maintain class distribution
        )

        # Second split: separate train and validation
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )

        print(f"   Train set:      {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

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
        print("⚖️ Scaling features...")

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
        print(f"🌲 Training Random Forest with {self.model_params['n_estimators']} trees...")

        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

        print("✅ Model training complete")

        return self.model

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
        print(f"\n📈 Evaluating on {dataset_name} set...")

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
        Perform cross-validation

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            CV scores
        """
        print(f"\n🔄 Performing {self.cv_folds}-fold cross-validation...")

        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=self.cv_folds,
            scoring='accuracy'
        )

        print(f"   CV Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

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

        print("\n🔍 Feature Importance:")
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
        print(f"💾 Saved model to {model_path}")

        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")

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

        print(f"💾 Saved metadata to {metadata_path}")

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

        print(f"📂 Loaded model from {model_path}")
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
        print("🎯 Starting ML Model Training Pipeline")
        print("=" * 60)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_test_split(
            df, feature_cols, target_col
        )

        # Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        # Train model
        self.train_random_forest(X_train, y_train)

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
        print("✅ Training Pipeline Complete!")
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
