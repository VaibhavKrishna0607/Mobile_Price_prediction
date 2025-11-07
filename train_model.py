"""
Training script for Mobile Price Predictor
This script trains a Random Forest Classifier on mobile phone specifications
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import argparse
from typing import Tuple, List


BASE_FEATURE_COLUMNS = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores',
    'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w',
    'talk_time', 'three_g', 'touch_screen', 'wifi'
]

ENGINEERED_FEATURE_COLUMNS = [
    'pixel_area', 'ppi', 'screen_ratio', 'ram_per_core', 'battery_per_weight'
]

def load_data(mob_price_classification_train: str) -> pd.DataFrame:
    """Load training data from CSV"""
    df = pd.read_csv(mob_price_classification_train)
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Preprocess the data and engineer additional features."""

    df = df.copy()

    missing_columns = [col for col in BASE_FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    # Feature engineering to boost model performance
    df['pixel_area'] = df['px_height'] * df['px_width']
    df['ppi'] = np.sqrt(df['px_height'] ** 2 + df['px_width'] ** 2) / np.where(df['sc_h'] == 0, 1, df['sc_h'])
    df['screen_ratio'] = np.where(df['sc_w'] == 0, 0, df['sc_h'] / df['sc_w'])
    df['ram_per_core'] = np.where(df['n_cores'] == 0, 0, df['ram'] / df['n_cores'])
    df['battery_per_weight'] = np.where(df['mobile_wt'] == 0, 0, df['battery_power'] / df['mobile_wt'])

    feature_columns = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

    target_column = 'price_range'
    if target_column not in df.columns:
        possible_targets = ['price', 'target', 'price_category']
        target_column = next((col for col in possible_targets if col in df.columns), None)
        if target_column is None:
            raise ValueError("Target column not found. Expected one of: price_range, price, target, price_category")

    X = df[feature_columns]
    y = df[target_column]

    return X, y, feature_columns

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = None,
    random_state: int = 42,
    tune: bool = False,
    tune_iter: int = 20
):
    """Train Random Forest model. Optionally run randomized hyperparameter search."""

    base_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )

    if not tune:
        base_model.fit(X_train, y_train)
        return base_model

    param_dist = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=tune_iter,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    search.fit(X_train, y_train)
    print(f"Best hyperparameters: {search.best_params_}")
    return search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, y_pred

def save_model(model, feature_columns, model_path='model.pkl'):
    """Save trained model together with metadata."""
    artifact = {
        'model': model,
        'feature_columns': feature_columns
    }
    joblib.dump(artifact, model_path)
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Mobile Price Predictor Model')
    parser.add_argument('--data', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--model-path', type=str, default='model.pkl', help='Path to save model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0-1)')
    parser.add_argument('--n-estimators', type=int, default=300, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=None, help='Max depth of trees (None for full depth)')
    parser.add_argument('--tune', action='store_true', help='Enable randomized hyperparameter search')
    parser.add_argument('--tune-iter', type=int, default=20, help='Number of parameter settings sampled when tuning')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_data(args.data)
    print(f"Data loaded: {df.shape}")
    
    # Preprocess
    print("Preprocessing data...")
    X, y, feature_columns = preprocess_data(df)
    print(f"Features: {X.shape}, Target: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    print("Training model...")
    model = train_model(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        tune=args.tune,
        tune_iter=args.tune_iter
    )
    
    # Evaluate
    print("Evaluating model...")
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, feature_columns, args.model_path)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()

