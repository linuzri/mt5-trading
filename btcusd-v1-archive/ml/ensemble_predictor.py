"""
Ensemble Model Prediction Module for ML Trading Strategy

This module handles:
- Loading 3 trained models (RF, XGBoost, LightGBM)
- Making predictions via majority voting (2/3 must agree)
- Providing confidence scores

Drop-in replacement for ModelPredictor with identical interface.
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
import warnings
from datetime import datetime

# Suppress sklearn feature name warnings (RF trained without names, LGB with names - both work fine)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class EnsemblePredictor:
    """Make predictions using ensemble of 3 ML models (RF + XGBoost + LightGBM)"""

    def __init__(self, config_path="ml_config.json"):
        """Initialize with ML configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Ensemble model paths
        ensemble_paths = self.config.get('ensemble_paths', {})
        self.rf_path = ensemble_paths.get('rf_model', 'models/ensemble_rf.pkl')
        self.xgb_path = ensemble_paths.get('xgb_model', 'models/ensemble_xgb.pkl')
        self.lgb_path = ensemble_paths.get('lgb_model', 'models/ensemble_lgb.pkl')
        self.scaler_path = ensemble_paths.get('scaler', 'models/ensemble_scaler.pkl')
        self.metadata_path = ensemble_paths.get('metadata', 'models/ensemble_metadata.json')

        self.confidence_threshold = self.config['prediction']['confidence_threshold']
        self.min_prob_diff = self.config['prediction']['min_probability_diff']
        self.max_hold_probability = self.config['prediction'].get('max_hold_probability', 0.50)

        self.models = {}  # {'rf': model, 'xgb': model, 'lgb': model}
        self.scaler = None
        self.feature_names = None
        self.label_map = {0: 'bad', 1: 'good'}  # Quality filter, not direction

    def load_model(self):
        """Load all 3 trained models and shared scaler"""
        model_files = {
            'rf': self.rf_path,
            'xgb': self.xgb_path,
            'lgb': self.lgb_path
        }

        for name, path in model_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Ensemble model not found: {path}\n"
                    f"Please train the ensemble first: python train_ml_model.py --ensemble --refresh"
                )
            print(f"[i] Loading {name.upper()} model from {path}")
            self.models[name] = joblib.load(path)

        self.scaler = joblib.load(self.scaler_path)

        # Load metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                print(f"   Features: {self.feature_names}")
                print(f"   Trained at: {metadata.get('trained_at')}")
                accuracies = metadata.get('individual_accuracies', {})
                for m, acc in accuracies.items():
                    print(f"   {m.upper()} accuracy: {acc:.4f}")
                print(f"   Ensemble accuracy: {metadata.get('ensemble_accuracy', 'N/A')}")
        else:
            self.feature_names = self.config['features']

        print("[OK] Ensemble models loaded successfully (RF + XGBoost + LightGBM)")

    def prepare_features(self, features_dict):
        """
        Prepare features for prediction

        Args:
            features_dict: Dictionary with feature names and values

        Returns:
            Scaled feature array ready for prediction
        """
        missing_features = set(self.feature_names) - set(features_dict.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        features = np.array([features_dict[f] for f in self.feature_names])
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return features_scaled

    def predict(self, features_dict, return_probabilities=True):
        """
        Make prediction using unanimous voting across all 3 models (3/3 must agree)

        Args:
            features_dict: Dictionary with feature names and values
            return_probabilities: Return probability scores

        Returns:
            prediction: 'buy', 'sell', or 'hold'
            confidence: Confidence score (0-1)
            probabilities: Dict of averaged probabilities for each class
        """
        if not self.models:
            self.load_model()

        features_scaled = self.prepare_features(features_dict)

        # Get predictions from all models
        all_probs = []
        for name, model in self.models.items():
            probs = model.predict_proba(features_scaled)[0]
            all_probs.append(probs)

        # Average probabilities across models
        avg_probs = np.mean(all_probs, axis=0)
        predicted_class = np.argmax(avg_probs)
        prediction = self.label_map[predicted_class]
        confidence = avg_probs[predicted_class]

        prob_dict = {
            'bad': avg_probs[0],
            'good': avg_probs[1],
        }

        if return_probabilities:
            return prediction, confidence, prob_dict
        else:
            return prediction, confidence

    def get_quality_signal(self, features_dict):
        """
        Quality filter: should we take this trend trade or skip it?

        Returns:
            ('take', confidence, reason) or ('skip', confidence, reason)
        """
        if not self.models:
            self.load_model()

        features_scaled = self.prepare_features(features_dict)

        # Get each model's prediction
        model_votes = {}
        model_good_probs = {}
        name_map = {'rf': 'RF', 'xgb': 'XGB', 'lgb': 'LGB'}

        for name, model in self.models.items():
            probs = model.predict_proba(features_scaled)[0]
            good_prob = probs[1]  # P(GOOD)
            model_good_probs[name] = good_prob
            model_votes[name] = 'GOOD' if good_prob > 0.5 else 'BAD'

        # Build vote description
        vote_parts = []
        for name in ['rf', 'xgb', 'lgb']:
            vote_parts.append(f"{name_map[name]}:{model_votes[name]}:{int(model_good_probs[name]*100)}%")
        votes_str = " ".join(vote_parts)

        # Count GOOD votes
        good_count = sum(1 for v in model_votes.values() if v == 'GOOD')

        # Average confidence of GOOD probability across all models
        avg_good_prob = np.mean(list(model_good_probs.values()))

        # TAKE: 2/3+ models predict GOOD AND averaged confidence > 65%
        quality_threshold = 0.65
        if good_count >= 2 and avg_good_prob > quality_threshold:
            reason = f"Quality: TAKE ({votes_str}) | {good_count}/3 GOOD | conf={avg_good_prob:.0%}"
            return 'take', avg_good_prob, reason
        else:
            if good_count < 2:
                reason = f"Quality: SKIP ({votes_str}) | only {good_count}/3 GOOD"
            else:
                reason = f"Quality: SKIP ({votes_str}) | conf={avg_good_prob:.0%} < {quality_threshold:.0%}"
            return 'skip', avg_good_prob, reason

    def _get_model_signal(self, probs):
        """Get signal from a single model's probabilities (binary: BAD=0, GOOD=1)"""
        bad_prob = probs[0]
        good_prob = probs[1]

        max_prob = max(good_prob, bad_prob)
        if max_prob < self.confidence_threshold:
            return 'hold', max_prob

        if good_prob > bad_prob:
            return 'good', good_prob
        else:
            return 'bad', bad_prob

    def get_trade_signal(self, features_dict):
        """
        Get actionable trade signal via majority voting (2/3 must agree)

        Args:
            features_dict: Dictionary with feature names and values

        Returns:
            signal: 'buy', 'sell', or None
            confidence: Confidence score
            reason: Explanation showing each model's vote
        """
        if not self.models:
            self.load_model()

        features_scaled = self.prepare_features(features_dict)

        # Get each model's prediction independently
        model_signals = {}
        model_confidences = {}
        model_probs = {}

        for name, model in self.models.items():
            probs = model.predict_proba(features_scaled)[0]
            model_probs[name] = probs
            signal, conf = self._get_model_signal(probs)
            model_signals[name] = signal
            model_confidences[name] = conf

        # Build vote description for each model
        name_map = {'rf': 'RF', 'xgb': 'XGB', 'lgb': 'LGB'}
        vote_parts = []
        for name in ['rf', 'xgb', 'lgb']:
            sig = model_signals[name].upper()
            conf_pct = int(model_confidences[name] * 100)
            vote_parts.append(f"{name_map[name]}:{sig}:{conf_pct}%")
        votes_str = " ".join(vote_parts)

        # Count votes
        from collections import Counter
        vote_counts = Counter(model_signals.values())
        most_common_signal, most_common_count = vote_counts.most_common(1)[0]

        # Majority vote: need 2/3 to agree
        if most_common_count >= 2 and most_common_signal != 'hold':
            signal = most_common_signal
            # Average confidence of agreeing models
            agreeing_confs = [model_confidences[n] for n in model_signals if model_signals[n] == signal]
            confidence = np.mean(agreeing_confs)

            # Apply thresholds
            if confidence < self.confidence_threshold:
                return None, confidence, f"Ensemble: {votes_str} | Confidence {confidence:.0%} below threshold"

            # Check prob diff using averaged probabilities
            avg_probs = np.mean([model_probs[n] for n in model_probs], axis=0)
            sell_prob = avg_probs[0]
            buy_prob = avg_probs[1]
            prob_diff = abs(buy_prob - sell_prob)
            if prob_diff < self.min_prob_diff:
                return None, confidence, f"Ensemble: {votes_str} | BUY/SELL diff {prob_diff:.0%} too small"

            reason = f"Ensemble: {signal.upper()} ({votes_str}) | {most_common_count}/3 agree"
            return signal, confidence, reason
        else:
            # No majority — models disagree = HOLD (no trade)
            return None, max(model_confidences.values()), f"Ensemble: NO CONSENSUS ({votes_str}) | models disagree"

    def batch_predict(self, features_df):
        """
        Make predictions on batch of data using averaged probabilities

        Args:
            features_df: DataFrame with feature columns

        Returns:
            predictions, confidences arrays
        """
        if not self.models:
            self.load_model()

        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # Average probabilities from all models
        all_probs = []
        for name, model in self.models.items():
            probs = model.predict_proba(X_scaled)
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        predictions = np.argmax(avg_probs, axis=1)
        confidences = np.max(avg_probs, axis=1)
        predictions_labeled = [self.label_map[p] for p in predictions]

        return predictions_labeled, confidences
