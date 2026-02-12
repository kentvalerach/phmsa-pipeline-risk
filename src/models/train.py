"""
PHMSA Pipeline Risk Model - Training Module
============================================

Walk-forward validated LightGBM classifier for pipeline incident prediction.

Usage:
    python -m src.models.train --data data/processed/survival_panel.csv

Author: Kent
Date: February 2026
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PipelineRiskModel:
    """
    LightGBM-based pipeline incident risk model with walk-forward validation.
    
    Attributes:
        model: Trained LightGBM classifier
        calibrator: Isotonic regression calibrator
        feature_names: List of feature column names
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self, params=None):
        """
        Initialize the model.
        
        Args:
            params: LightGBM hyperparameters (optional)
        """
        self.params = params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'reg_lambda': 1.0,
            'reg_alpha': 0.1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.metrics = {}
    
    def _prepare_features(self, df):
        """
        Prepare feature matrix from panel dataframe.
        
        Args:
            df: Panel dataframe with required columns
            
        Returns:
            X: Feature matrix (numpy array)
            feature_names: List of feature column names
        """
        # Core features
        df = df.copy()
        df['log_miles'] = np.log1p(df['miles_at_risk'])
        
        # Era dummies
        era_dummies = pd.get_dummies(df['era'], prefix='era', drop_first=True)
        
        # Feature columns
        feature_cols = [
            'log_miles',
            'age_at_obs',
            'pct_small_diam',
            'pct_large_diam',
            'pct_high_smys',
            'pct_class1'
        ]
        
        # Add optional enrichment features if present
        optional_cols = [
            'pct_low_smys',
            'pct_high_class',
            'log1p_total_repairs',
            'log1p_cum_corrosion',
            'lag_repairs_cl12',
            'soil_corr_index',
            'earthquake_count'
        ]
        
        for col in optional_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        X = np.column_stack([
            df[feature_cols].fillna(0).values,
            era_dummies.values
        ])
        
        feature_names = feature_cols + list(era_dummies.columns)
        
        return X, feature_names
    
    def train(self, df, train_mask, val_mask=None):
        """
        Train the model.
        
        Args:
            df: Panel dataframe
            train_mask: Boolean mask for training rows
            val_mask: Boolean mask for validation rows (optional)
            
        Returns:
            self
        """
        logger.info("Preparing features...")
        X, self.feature_names = self._prepare_features(df)
        y = df['event'].values
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        logger.info(f"Training set: {len(X_train):,} samples, {y_train.sum():.0f} events")
        
        # Create LightGBM model
        self.model = lgb.LGBMClassifier(**self.params)
        
        if val_mask is not None:
            X_val = X[val_mask]
            y_val = y[val_mask]
            logger.info(f"Validation set: {len(X_val):,} samples, {y_val.sum():.0f} events")
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        logger.info("Training complete")
        
        return self
    
    def calibrate(self, df, cal_mask):
        """
        Calibrate predicted probabilities using isotonic regression.
        
        Args:
            df: Panel dataframe
            cal_mask: Boolean mask for calibration rows
            
        Returns:
            self
        """
        logger.info("Calibrating probabilities...")
        
        X, _ = self._prepare_features(df)
        y = df['event'].values
        
        X_cal = X[cal_mask]
        y_cal = y[cal_mask]
        
        # Get raw predictions
        raw_probs = self.model.predict_proba(X_cal)[:, 1]
        
        # Fit calibrator
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probs, y_cal)
        
        logger.info("Calibration complete")
        
        return self
    
    def predict_proba(self, df):
        """
        Predict calibrated probabilities.
        
        Args:
            df: Panel dataframe
            
        Returns:
            Calibrated probability array
        """
        X, _ = self._prepare_features(df)
        raw_probs = self.model.predict_proba(X)[:, 1]
        
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs)
        else:
            return raw_probs
    
    def evaluate(self, df, test_mask, name="test"):
        """
        Evaluate model performance.
        
        Args:
            df: Panel dataframe
            test_mask: Boolean mask for test rows
            name: Name for this evaluation set
            
        Returns:
            Dictionary of metrics
        """
        X, _ = self._prepare_features(df)
        y = df['event'].values
        
        y_test = y[test_mask]
        y_prob = self.predict_proba(df.loc[test_mask])
        
        metrics = {
            f'{name}_auc': roc_auc_score(y_test, y_prob),
            f'{name}_brier': brier_score_loss(y_test, y_prob),
            f'{name}_n': len(y_test),
            f'{name}_events': int(y_test.sum())
        }
        
        self.metrics.update(metrics)
        
        logger.info(f"{name.upper()} - AUC: {metrics[f'{name}_auc']:.4f}, "
                   f"Brier: {metrics[f'{name}_brier']:.4f}")
        
        return metrics
    
    def walk_forward_cv(self, df, train_end_years=[2015, 2016, 2017, 2018, 2019]):
        """
        Perform walk-forward cross-validation.
        
        Args:
            df: Panel dataframe
            train_end_years: List of training end years
            
        Returns:
            DataFrame with CV results
        """
        results = []
        
        for i, train_end in enumerate(train_end_years):
            test_start = train_end + 1
            
            logger.info(f"\nFold {i+1}: Train ≤{train_end}, Test {test_start}")
            
            train_mask = df['year'] <= train_end
            test_mask = df['year'] == test_start
            
            if i == len(train_end_years) - 1:
                # Last fold: test on remaining years
                test_mask = df['year'] >= test_start
            
            # Train model
            fold_model = PipelineRiskModel(self.params)
            fold_model.train(df, train_mask)
            
            # Evaluate
            X, _ = fold_model._prepare_features(df)
            y = df['event'].values
            y_test = y[test_mask]
            y_prob = fold_model.model.predict_proba(X[test_mask])[:, 1]
            
            auc = roc_auc_score(y_test, y_prob)
            
            results.append({
                'fold': i + 1,
                'train_end': train_end,
                'test_period': f"{test_start}+" if i == len(train_end_years) - 1 else str(test_start),
                'train_n': train_mask.sum(),
                'test_n': test_mask.sum(),
                'test_events': int(y_test.sum()),
                'auc': auc
            })
            
            logger.info(f"  AUC: {auc:.4f}")
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance['importance_pct'] = (
            importance['importance'] / importance['importance'].sum() * 100
        )
        
        return importance
    
    def save(self, path):
        """
        Save model artifacts.
        
        Args:
            path: Directory path for saving
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, path / 'lgbm_model.pkl')
        if self.calibrator is not None:
            joblib.dump(self.calibrator, path / 'calibrator.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'params': self.params,
            'metrics': self.metrics,
            'trained_at': datetime.now().isoformat()
        }
        pd.Series(metadata).to_json(path / 'metadata.json')
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load model from saved artifacts.
        
        Args:
            path: Directory path containing saved model
            
        Returns:
            PipelineRiskModel instance
        """
        path = Path(path)
        
        model = cls()
        model.model = joblib.load(path / 'lgbm_model.pkl')
        
        if (path / 'calibrator.pkl').exists():
            model.calibrator = joblib.load(path / 'calibrator.pkl')
        
        metadata = pd.read_json(path / 'metadata.json', typ='series')
        model.feature_names = metadata['feature_names']
        model.params = metadata['params']
        model.metrics = metadata['metrics']
        
        return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train pipeline risk model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to survival panel CSV')
    parser.add_argument('--output', type=str, default='models/',
                       help='Output directory for model artifacts')
    parser.add_argument('--train-end', type=int, default=2019,
                       help='Last year of training data')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PHMSA Pipeline Risk Model - Training")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df):,} rows, {df['event'].sum():.0f} events")
    
    # Define splits
    train_mask = df['year'] <= args.train_end
    test_mask = df['year'] > args.train_end
    
    # Train model
    model = PipelineRiskModel()
    model.train(df, train_mask)
    model.calibrate(df, train_mask)
    
    # Evaluate
    model.evaluate(df, train_mask, name="train")
    model.evaluate(df, test_mask, name="test")
    
    # Walk-forward CV
    logger.info("\nWalk-forward cross-validation:")
    cv_results = model.walk_forward_cv(df)
    print(cv_results.to_string(index=False))
    
    # Feature importance
    logger.info("\nFeature importance:")
    importance = model.get_feature_importance()
    print(importance.head(10).to_string(index=False))
    
    # Save
    model.save(args.output)
    cv_results.to_csv(Path(args.output) / 'cv_results.csv', index=False)
    importance.to_csv(Path(args.output) / 'feature_importance.csv', index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
