from flask import Flask, render_template, request, jsonify
import boto3
import json
import os
from datetime import datetime
import logging
import numpy as np
import joblib
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    raise RuntimeError("SECRET_KEY environment variable is not set. Add it to your .env file.")
app.config['SECRET_KEY'] = secret_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
SAGEMAKER_ENDPOINT = os.environ.get('SAGEMAKER_ENDPOINT', 'mobile-price-predictor-endpoint')
SAGEMAKER_ROLE = os.environ.get('SAGEMAKER_ROLE', 'arn:aws:iam::your-account:role/SageMakerRole')

# Check if AWS credentials are available
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
USE_MOCK_PREDICTIONS = os.environ.get('USE_MOCK_PREDICTIONS', 'false').lower() == 'true'

# Load local model if available
LOCAL_MODEL_PATH = os.environ.get('LOCAL_MODEL_PATH', 'mobile_price_model.pkl')
local_model_artifact = None
try:
    if os.path.exists(LOCAL_MODEL_PATH):
        local_model_artifact = joblib.load(LOCAL_MODEL_PATH)
        logger.info(f"Local model loaded from {LOCAL_MODEL_PATH}")
    else:
        logger.warning(f"Local model not found at {LOCAL_MODEL_PATH}. Using mock predictions.")
except Exception as e:
    logger.warning(f"Failed to load local model: {e}")

# Load real-world smartphones dataset for recommendations
import pandas as pd
REAL_PHONES_PATH = os.environ.get('REAL_PHONES_PATH', 'real_world_smartphones.csv')
real_phones_df = None
try:
    if os.path.exists(REAL_PHONES_PATH):
        real_phones_df = pd.read_csv(REAL_PHONES_PATH)
        real_phones_df = real_phones_df.dropna(subset=['price', 'ram_capacity', 'battery_capacity', 'primary_camera_rear', 'internal_memory'])
        logger.info(f"Real phones dataset loaded: {len(real_phones_df)} phones")
    else:
        logger.warning(f"Real phones dataset not found at {REAL_PHONES_PATH}")
except Exception as e:
    logger.warning(f"Failed to load real phones dataset: {e}")

# Initialize SageMaker clients (only if credentials are available)
sagemaker_runtime = None
sagemaker_client = None

if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and not USE_MOCK_PREDICTIONS:
    try:
        sagemaker_runtime = boto3.client(
            'sagemaker-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        sagemaker_client = boto3.client(
            'sagemaker',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        logger.info("AWS SageMaker clients initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize AWS clients: {e}")
        USE_MOCK_PREDICTIONS = True
else:
    if USE_MOCK_PREDICTIONS:
        logger.info("Using mock predictions (USE_MOCK_PREDICTIONS=true)")
    else:
        logger.warning("AWS credentials not found. Using mock predictions for testing.")
        USE_MOCK_PREDICTIONS = True


FORM_FIELDS = [
    'ram_capacity', 'battery_capacity', 'processor_speed', 'num_cores',
    'primary_camera_rear', 'primary_camera_front', 'internal_memory',
    'resolution_height', 'resolution_width', 'screen_size', 'refresh_rate',
    '5G_or_not', 'fast_charging_available', 'extended_memory_available', 'num_rear_cameras'
]

_FIELD_CONVERTERS = {
    'ram_capacity': float, 'battery_capacity': float, 'processor_speed': float,
    'num_cores': int, 'primary_camera_rear': int, 'primary_camera_front': int,
    'internal_memory': int, 'resolution_height': int, 'resolution_width': int,
    'screen_size': float, 'refresh_rate': int, '5G_or_not': int,
    'fast_charging_available': int, 'extended_memory_available': int, 'num_rear_cameras': int,
}


def prepare_features(form_data):
    """Extract and type-convert form fields into the feature vector for the model."""
    return [_FIELD_CONVERTERS[f](form_data.get(f) or '0') for f in FORM_FIELDS]


BASE_FEATURE_COLUMNS = FORM_FIELDS


def local_predict_price_range(features):
    """Predict using the locally trained .pkl model with feature engineering."""
    model = local_model_artifact['model']

    # features order: ram(0), battery(1), proc_speed(2), num_cores(3),
    # pc_rear(4), pc_front(5), int_mem(6), res_h(7), res_w(8), screen_size(9),
    # refresh_rate(10), 5G(11), fast_charging(12), ext_mem(13), num_cams(14)
    ram, battery, _, num_cores, _, _, _, res_h, res_w, screen_size, _, _, _, _, _ = features

    pixel_area = res_h * res_w
    ppi = np.sqrt(res_h ** 2 + res_w ** 2) / (screen_size if screen_size > 0 else 1)
    ram_per_core = ram / num_cores if num_cores > 0 else 0
    battery_per_ram = battery / ram if ram > 0 else 0

    full_features = list(features) + [pixel_area, ppi, ram_per_core, battery_per_ram]
    prediction = model.predict([full_features])[0]
    return {'predictions': [{'predicted_label': int(prediction)}]}


def mock_predict_price_range(features):
    """Fallback rule-based prediction when no model is available."""
    ram_gb  = features[0]   # GB
    battery = features[1]   # mAh
    pc      = features[4]   # primary camera MP
    res_h   = features[7]   # resolution height px
    five_g  = features[11]  # 0/1

    score = 0.0
    if ram_gb >= 12:      score += 3
    elif ram_gb >= 8:     score += 2
    elif ram_gb >= 6:     score += 1.5
    elif ram_gb >= 4:     score += 0.5

    if battery >= 5000:   score += 0.5
    elif battery >= 4000: score += 0.3

    if pc >= 108:         score += 2
    elif pc >= 50:        score += 1
    elif pc >= 12:        score += 0.5

    if res_h >= 2400:     score += 1
    elif res_h >= 1080:   score += 0.5

    if five_g:            score += 1

    if score >= 6:        prediction = 3
    elif score >= 4:      prediction = 2
    elif score >= 2:      prediction = 1
    else:                 prediction = 0

    return {'predictions': [{'predicted_label': prediction}]}


def recommend_phones(features, predicted_range, n=5):
    """
    Find top N real-world phones matching the predicted price range and user specs.
    Uses weighted similarity scoring with brand diversity to show variety.
    """
    if real_phones_df is None:
        return []

    price_bins = {
        0: (0, 10000),
        1: (10000, 20000),
        2: (20000, 35000),
        3: (35000, float('inf'))
    }

    low, high = price_bins.get(predicted_range, (0, float('inf')))
    candidates = real_phones_df[
        (real_phones_df['price'] >= low) & (real_phones_df['price'] < high)
    ].copy()

    if candidates.empty:
        return []

    # User specs from feature vector (matches prepare_features order)
    ram_gb   = features[0]   # GB directly
    battery  = features[1]   # mAh
    pc       = features[4]   # primary camera MP
    int_mem  = features[6]   # internal memory GB

    # Normalised absolute difference per feature (0 = perfect match, 1 = worst)
    def norm_diff(user_val, col):
        col_min = candidates[col].min()
        col_max = candidates[col].max()
        spread = col_max - col_min
        if spread == 0:
            return 0.0
        return (candidates[col] - user_val).abs() / spread

    candidates['_score'] = (
        norm_diff(ram_gb, 'ram_capacity')        * 0.35 +
        norm_diff(battery, 'battery_capacity')   * 0.25 +
        norm_diff(pc, 'primary_camera_rear')     * 0.20 +
        norm_diff(int_mem, 'internal_memory')    * 0.20
    )

    # Get top matches with brand diversity
    # First, get more candidates than needed
    top_candidates = candidates.nsmallest(n * 3, '_score')
    
    # Select diverse brands - max 2 phones per brand
    selected = []
    brand_count = {}
    
    for _, phone in top_candidates.iterrows():
        brand = phone['brand_name']
        if brand_count.get(brand, 0) < 2:  # Max 2 per brand
            selected.append(phone)
            brand_count[brand] = brand_count.get(brand, 0) + 1
            if len(selected) >= n:
                break
    
    # If we don't have enough, fill with remaining best matches
    if len(selected) < n:
        remaining = top_candidates[~top_candidates.index.isin([p.name for p in selected])]
        for _, phone in remaining.iterrows():
            selected.append(phone)
            if len(selected) >= n:
                break
    
    # Convert to DataFrame for easier processing
    result_df = pd.DataFrame(selected)[[
        'brand_name', 'model', 'price', 'avg_rating',
        'ram_capacity', 'battery_capacity', 'primary_camera_rear',
        'internal_memory', '5G_or_not', 'refresh_rate'
    ]].copy()

    result_df['brand_name'] = result_df['brand_name'].str.title()
    result_df['5G_or_not'] = result_df['5G_or_not'].astype(int)

    return result_df.to_dict('records')


def predict_price_range(features):
    """
    Call SageMaker endpoint to get price prediction.
    Falls back to local model, then mock if neither is available.
    """
    if USE_MOCK_PREDICTIONS or sagemaker_runtime is None:
        if local_model_artifact is not None:
            logger.info("Using local model prediction")
            return local_predict_price_range(features)
        logger.info("Using mock prediction (no AWS, no local model)")
        return mock_predict_price_range(features)
    
    try:
        # Format data for SageMaker endpoint (CSV format)
        payload = ','.join(map(str, features))
        
        logger.info(f"Calling SageMaker endpoint: {SAGEMAKER_ENDPOINT}")
        logger.info(f"Payload: {payload}")
        
        # Invoke the endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='text/csv',
            Body=payload
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        logger.info(f"Prediction result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calling SageMaker endpoint: {str(e)}")
        logger.info("Falling back to mock prediction")
        return mock_predict_price_range(features)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        form_data = request.json if request.is_json else request.form
        
        # Prepare features
        features = prepare_features(form_data)
        
        # Validate: all required fields must be provided
        missing = [f for f in FORM_FIELDS if not form_data.get(f, '')]
        if missing:
            return jsonify({
                'success': False,
                'error': 'Please fill in all fields'
            }), 400
        
        # Get prediction from SageMaker
        prediction_result = predict_price_range(features)
        
        # Map prediction to price range labels
        price_ranges = {
            0: 'Budget mobile phone (0-10000)',
            1: 'Lower mid-range phone (10000-20000)',
            2: 'Upper mid-range phone (20000-35000)',
            3: 'Premium phone (35000+)'
        }
        
        # Extract prediction (assuming result contains 'predictions' or direct value)
        if isinstance(prediction_result, dict):
            prediction = prediction_result.get('predictions', [{}])[0].get('predicted_label', 0)
        else:
            prediction = int(prediction_result) if isinstance(prediction_result, (int, float)) else 0
        
        price_range = price_ranges.get(prediction, 'Unknown')

        # Get real phone recommendations - show more for variety (8 phones)
        recommended_phones = recommend_phones(features, int(prediction), n=8)

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'price_range': price_range,
            'features': features,
            'recommended_phones': recommended_phones
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if USE_MOCK_PREDICTIONS or sagemaker_client is None:
        return jsonify({
            'status': 'healthy',
            'mode': 'mock',
            'message': 'Running in mock mode (AWS not configured)',
            'timestamp': datetime.now().isoformat()
        })
    
    try:
        # Check if SageMaker endpoint is available
        endpoint_status = sagemaker_client.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT)
        status = endpoint_status['EndpointStatus']
        
        return jsonify({
            'status': 'healthy' if status == 'InService' else 'degraded',
            'mode': 'aws',
            'endpoint_status': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'degraded',
            'mode': 'aws',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503


@app.route('/endpoint-status', methods=['GET'])
def endpoint_status():
    """Check SageMaker endpoint status"""
    if USE_MOCK_PREDICTIONS or sagemaker_client is None:
        return jsonify({
            'success': True,
            'mode': 'mock',
            'message': 'Running in mock mode. AWS SageMaker endpoint not configured.',
            'endpoint_name': 'N/A',
            'status': 'Mock Mode'
        })
    
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT)
        return jsonify({
            'success': True,
            'mode': 'aws',
            'endpoint_name': response['EndpointName'],
            'status': response['EndpointStatus'],
            'creation_time': response['CreationTime'].isoformat() if 'CreationTime' in response else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'mode': 'aws',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Check if running in development mode
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 5000))
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

