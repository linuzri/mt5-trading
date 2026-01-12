"""
Model Prediction Module for ML Trading Strategy

This module handles:
- Loading trained models
- Making live predictions on current market data
- Providing confidence scores
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from datetime import datetime


class ModelPredictor:
    """Make predictions using trained ML model"""

    def __init__(self, config_path="ml_config.json"):
        """Initialize with ML configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.model_path = self.config['paths']['model_file']
        self.scaler_path = self.config['paths']['scaler_file']
        self.confidence_threshold = self.config['prediction']['confidence_threshold']
        self.min_prob_diff = self.config['prediction']['min_probability_diff']

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_map = {0: 'sell', 1: 'buy', 2: 'hold'}

    def load_model(self):
        """Load trained model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please train the model first by running: python train_ml_model.py"
            )

        print(f"[i] Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        # Load metadata to get feature names
        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                print(f"   Features: {self.feature_names}")
                print(f"   Trained at: {metadata.get('trained_at')}")
        else:
            # Fallback to config
            self.feature_names = self.config['features']

        print("[OK] Model loaded successfully")

    def prepare_features(self, features_dict):
        """
        Prepare features for prediction

        Args:
            features_dict: Dictionary with feature names and values

        Returns:
            Scaled feature array ready for prediction
        """
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(features_dict.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Extract features in correct order
        features = np.array([features_dict[f] for f in self.feature_names])

        # Reshape for single prediction
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        return features_scaled

    def predict(self, features_dict, return_probabilities=True):
        """
        Make prediction on current market state

        Args:
            features_dict: Dictionary with feature names and values
            return_probabilities: Return probability scores

        Returns:
            prediction: 'buy', 'sell', or 'hold'
            confidence: Confidence score (0-1)
            probabilities: Dict of probabilities for each class (optional)
        """
        if self.model is None:
            self.load_model()

        # Prepare features
        features_scaled = self.prepare_features(features_dict)

        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Get predicted class
        predicted_class = np.argmax(probabilities)
        prediction = self.label_map[predicted_class]
        confidence = probabilities[predicted_class]

        # Create probability dict
        prob_dict = {
            'sell': probabilities[0],
            'buy': probabilities[1],
            'hold': probabilities[2]
        }

        if return_probabilities:
            return prediction, confidence, prob_dict
        else:
            return prediction, confidence

    def get_trade_signal(self, features_dict):
        """
        Get actionable trade signal with confidence filtering

        Args:
            features_dict: Dictionary with feature names and values

        Returns:
            signal: 'buy', 'sell', or None (if confidence too low)
            confidence: Confidence score
            reason: Explanation of decision
        """
        prediction, confidence, probabilities = self.predict(features_dict)

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None, confidence, f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.2%}"

        # Check probability difference (avoid near-ties)
        sorted_probs = sorted(probabilities.values(), reverse=True)
        prob_diff = sorted_probs[0] - sorted_probs[1]

        if prob_diff < self.min_prob_diff:
            return None, confidence, f"Probability difference {prob_diff:.2%} too small"

        # Don't trade on HOLD signal
        if prediction == 'hold':
            return None, confidence, "Model predicts HOLD"

        # Valid trade signal
        reason = f"Model: {prediction.upper()} with {confidence:.2%} confidence"
        return prediction, confidence, reason

    def batch_predict(self, features_df):
        """
        Make predictions on batch of data

        Args:
            features_df: DataFrame with feature columns

        Returns:
            predictions, confidences arrays
        """
        if self.model is None:
            self.load_model()

        # Extract features in correct order
        X = features_df[self.feature_names].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Get confidence (max probability for each prediction)
        confidences = np.max(probabilities, axis=1)

        # Map predictions to labels
        predictions_labeled = [self.label_map[p] for p in predictions]

        return predictions_labeled, confidences

    def backtest_predictions(self, df_with_features):
        """
        Backtest model predictions on historical data

        Args:
            df_with_features: DataFrame with features and actual labels

        Returns:
            DataFrame with predictions and results
        """
        print("[i] Backtesting model predictions...")

        # Make predictions
        predictions, confidences = self.batch_predict(df_with_features)

        # Add to dataframe
        df_results = df_with_features.copy()
        df_results['ml_prediction'] = predictions
        df_results['ml_confidence'] = confidences

        # Calculate accuracy if labels are present
        if 'label' in df_results.columns:
            # Map label numbers to names for comparison
            df_results['actual_label'] = df_results['label'].map(self.label_map)
            df_results['correct'] = df_results['ml_prediction'] == df_results['actual_label']

            accuracy = df_results['correct'].mean()
            print(f"   Backtest Accuracy: {accuracy:.2%}")

            # Filter by confidence threshold
            high_confidence = df_results[df_results['ml_confidence'] >= self.confidence_threshold]
            if len(high_confidence) > 0:
                high_conf_accuracy = high_confidence['correct'].mean()
                print(f"   High-Confidence Accuracy: {high_conf_accuracy:.2%} ({len(high_confidence)} predictions)")

        return df_results


if __name__ == "__main__":
    # Test model predictor
    print("=" * 60)
    print("ML Model Prediction Test")
    print("=" * 60)

    predictor = ModelPredictor("../ml_config.json")

    # Try to load model
    try:
        predictor.load_model()

        # Test with dummy features
        test_features = {
            'rsi_14': 45.5,
            'macd_line': 0.002,
            'macd_signal': 0.001,
            'atr_14': 50.0,
            'bb_upper': 45000,
            'bb_lower': 44000,
            'bb_width': 0.02,
            'volume_ratio': 1.2,
            'price_change_1min': 0.001,
            'price_change_5min': 0.003
        }

        print("\nTest Features:")
        for k, v in test_features.items():
            print(f"   {k}: {v}")

        # Get prediction
        prediction, confidence, probabilities = predictor.predict(test_features)

        print(f"\n[>] Prediction: {prediction.upper()}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Probabilities:")
        for label, prob in probabilities.items():
            print(f"      {label}: {prob:.2%}")

        # Get trade signal
        signal, conf, reason = predictor.get_trade_signal(test_features)
        print(f"\n[i] Trade Signal: {signal}")
        print(f"   Reason: {reason}")

    except FileNotFoundError as e:
        print(f"\n[!] {e}")
        print("   Please train the model first!")
