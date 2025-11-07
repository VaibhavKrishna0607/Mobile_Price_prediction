from flask import Flask, render_template, request, jsonify
import boto3
import json
import os
from datetime import datetime
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

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


def prepare_features(form_data):
    """
    Prepare features from form data to match model input format.
    Maps form fields to the expected feature array.
    """
    # Feature order (20 features):
    # battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
    # m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w,
    # talk_time, three_g, touch_screen, wifi
    
    features = [
        int(form_data.get('battery_power', 0)),
        int(form_data.get('blue', 0)),
        float(form_data.get('clock_speed', 0)),
        int(form_data.get('dual_sim', 0)),
        int(form_data.get('fc', 0)),  # Front Camera megapixels
        int(form_data.get('four_g', 0)),
        int(form_data.get('int_memory', 0)),  # Internal Memory in GB
        float(form_data.get('m_dep', 0)),  # Mobile Depth in cm
        int(form_data.get('mobile_wt', 0)),  # Weight in grams
        int(form_data.get('n_cores', 0)),
        int(form_data.get('pc', 0)),  # Primary Camera megapixels
        int(form_data.get('px_height', 0)),  # Pixel Resolution Height
        int(form_data.get('px_width', 0)),  # Pixel Resolution Width
        int(form_data.get('ram', 0)),  # RAM in MB
        int(form_data.get('sc_h', 0)),  # Screen Height in cm
        int(form_data.get('sc_w', 0)),  # Screen Width in cm
        int(form_data.get('talk_time', 0)),  # Talk time in hours
        int(form_data.get('three_g', 0)),
        int(form_data.get('touch_screen', 0)),
        int(form_data.get('wifi', 0))
    ]
    
    return features


def mock_predict_price_range(features):
    """
    Mock prediction function for testing without AWS
    Simple rule-based prediction based on key features
    """
    ram = features[13]  # RAM in MB
    battery = features[0]  # Battery power
    pc = features[10]  # Primary camera
    px_height = features[11]  # Pixel height
    px_width = features[12]  # Pixel width
    
    # Simple rule-based classification
    score = 0
    
    # RAM contribution
    if ram >= 6144:  # 6GB+
        score += 3
    elif ram >= 4096:  # 4GB+
        score += 2
    elif ram >= 3072:  # 3GB+
        score += 1
    
    # Battery contribution
    if battery >= 4000:
        score += 1
    elif battery >= 3000:
        score += 0.5
    
    # Camera contribution
    if pc >= 48:
        score += 2
    elif pc >= 24:
        score += 1
    elif pc >= 12:
        score += 0.5
    
    # Display resolution
    if px_height >= 2400 and px_width >= 1080:
        score += 1
    elif px_height >= 1920:
        score += 0.5
    
    # Map score to price range (0-3)
    if score >= 6:
        prediction = 3  # Premium
    elif score >= 4:
        prediction = 2  # Upper mid-range
    elif score >= 2:
        prediction = 1  # Lower mid-range
    else:
        prediction = 0  # Budget
    
    return {'predictions': [{'predicted_label': prediction}]}


def predict_price_range(features):
    """
    Call SageMaker endpoint to get price prediction
    Falls back to mock prediction if AWS is not available
    """
    if USE_MOCK_PREDICTIONS or sagemaker_runtime is None:
        logger.info("Using mock prediction (AWS not configured)")
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
        
        # Validate features
        if not all(features):
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
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'price_range': price_range,
            'features': features
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

