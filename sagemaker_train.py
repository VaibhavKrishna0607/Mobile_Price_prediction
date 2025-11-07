"""
SageMaker Training Script
This script is used by AWS SageMaker to train the model
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json


BASE_FEATURE_COLUMNS = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores',
    'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w',
    'talk_time', 'three_g', 'touch_screen', 'wifi'
]

ENGINEERED_FEATURE_COLUMNS = [
    'pixel_area', 'ppi', 'screen_ratio', 'ram_per_core', 'battery_per_weight'
]

ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_columns = [col for col in BASE_FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    df['pixel_area'] = df['px_height'] * df['px_width']
    df['ppi'] = np.sqrt(df['px_height'] ** 2 + df['px_width'] ** 2) / np.where(df['sc_h'] == 0, 1, df['sc_h'])
    df['screen_ratio'] = np.where(df['sc_w'] == 0, 0, df['sc_h'] / df['sc_w'])
    df['ram_per_core'] = np.where(df['n_cores'] == 0, 0, df['ram'] / df['n_cores'])
    df['battery_per_weight'] = np.where(df['mobile_wt'] == 0, 0, df['battery_power'] / df['mobile_wt'])

    return df

# SageMaker reads the input data from specific environment variables
def model_fn(model_dir):
    """Load the model artifact from the model_dir directory"""
    artifact = joblib.load(os.path.join(model_dir, 'model.pkl'))
    if isinstance(artifact, dict) and 'model' in artifact:
        return artifact
    # Backwards compatibility for older artifacts
    return {
        'model': artifact,
        'feature_columns': ALL_FEATURE_COLUMNS
    }

def input_fn(request_body, request_content_type):
    """Deserialize and prepare the prediction input"""
    if request_content_type == 'text/csv':
        values = [float(x) for x in request_body.rstrip().split(',') if x]
        return np.array(values, dtype=float).reshape(1, -1)
    raise ValueError(f"Content type {request_content_type} not supported")

def predict_fn(input_data, artifact):
    """Perform prediction on the deserialized input"""
    model = artifact['model']
    feature_columns = artifact['feature_columns']

    if input_data.shape[1] == len(BASE_FEATURE_COLUMNS):
        df = pd.DataFrame(input_data, columns=BASE_FEATURE_COLUMNS)
        df = engineer_features(df)
        data = df[feature_columns]
    else:
        # Assume engineered features already provided
        data = pd.DataFrame(input_data, columns=feature_columns)

    prediction = model.predict(data)
    return prediction

def output_fn(prediction, content_type):
    """Serialize the prediction result"""
    if content_type == 'application/json':
        return json.dumps({
            'predictions': [
                {'predicted_label': int(pred)}
                for pred in prediction
            ]
        })
    elif content_type == 'text/csv':
        return ','.join([str(int(pred)) for pred in prediction])
    else:
        raise ValueError(f"Content type {content_type} not supported")

if __name__ == '__main__':
    # Training code (executed during SageMaker training job)
    # SageMaker sets environment variables: SM_MODEL_DIR, SM_CHANNEL_TRAINING
    
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    training_data_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    
    # Load training data
    train_file = os.path.join(training_data_dir, 'train.csv')
    df = pd.read_csv(train_file)
    
    # Feature columns
    df = engineer_features(df)

    feature_columns = ALL_FEATURE_COLUMNS
    
    target_column = 'price_range'
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, 'model.pkl')
    artifact = {
        'model': model,
        'feature_columns': feature_columns
    }
    joblib.dump(artifact, model_path)
    print(f"Model saved to {model_path}")

