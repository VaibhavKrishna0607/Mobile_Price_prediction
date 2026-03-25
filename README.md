# 📱 Smartphone Price Predictor

> ⚠️ **ALERT:** This application has been migrated from Render to Replit. The Render deployment has expired.

## 🚀 **Live Demo — Now Deployed on Replit!**
### 👉 **[https://mobi-predict--vaibhavkrishn15.replit.app/](https://mobi-predict--vaibhavkrishn15.replit.app/)**

A web application that predicts smartphone price ranges using a machine learning model deployed on AWS SageMaker. This application classifies smartphones into 4 price categories based on 20 different specifications.

## ✨ Features

- **Price Range Prediction**: Predicts smartphone price in 4 categories:
  - 💰 Budget mobile phone (₹0 - ₹10,000)
  - 📱 Lower mid-range phone (₹10,000 - ₹20,000)
  - 🎯 Upper mid-range phone (₹20,000 - ₹35,000)
  - 💎 Premium phone (₹35,000+)

- **Interactive Web Interface**: Modern, responsive design that works on desktop and mobile devices
- **Real-time Predictions**: Get instant predictions using AWS SageMaker endpoint
- **Comprehensive Input**: Handles 20 different smartphone specifications for accurate prediction

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla JS)
- **Backend**: Flask (Python)
- **ML Framework**: Scikit-learn (Random Forest Classifier)
- **Cloud**: AWS SageMaker, AWS S3, AWS IAM
- **AWS Services**: 
  - AWS SageMaker (Model training & deployment)
  - AWS S3 (Data storage)
  - AWS IAM (Access management)

## 📋 Prerequisites

- Python 3.8 or higher
- AWS Account with appropriate permissions
- AWS CLI configured (optional, for command-line operations)
- pip (Python package manager)

## 🚀 Installation

### 1. Clone or Navigate to Project Directory

```bash
cd mobile-price-predictor
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
SAGEMAKER_ENDPOINT=mobile-price-predictor-endpoint
SAGEMAKER_ROLE=arn:aws:iam::your-account-id:role/SageMakerExecutionRole
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
PORT=5000
S3_BUCKET_NAME=your-bucket-name
```

## 🔧 AWS Setup Guide

### Step 1: Create AWS Account

If you don't have an AWS account, create one at [aws.amazon.com](https://aws.amazon.com)

### Step 2: Set Up IAM Role for SageMaker

1. **Go to IAM Console**:
   - Navigate to AWS IAM Console
   - Click on "Roles" → "Create role"

2. **Select Service**:
   - Choose "SageMaker" as the trusted entity
   - Click "Next"

3. **Attach Policies**:
   - Attach `AmazonSageMakerFullAccess`
   - Attach `AmazonS3FullAccess` (or create a custom policy for specific buckets)
   - Click "Next"

4. **Name the Role**:
   - Role name: `SageMakerExecutionRole`
   - Click "Create role"

5. **Copy Role ARN**:
   - After creation, copy the Role ARN
   - Format: `arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole`
   - Add this to your `.env` file

### Step 3: Create S3 Bucket

1. **Go to S3 Console**:
   - Navigate to AWS S3 Console
   - Click "Create bucket"

2. **Configure Bucket**:
   - Bucket name: `mobile-price-predictor-<your-name>` (must be globally unique)
   - Region: Choose your preferred region (e.g., `us-east-1`)
   - Keep other settings as default
   - Click "Create bucket"

3. **Update `.env`**:
   - Set `S3_BUCKET_NAME` to your bucket name

### Step 4: Prepare Training Data

Your training data should be a CSV file with the following columns:

**Features (20 columns)**:
- `battery_power`: Battery capacity in mAh
- `blue`: Bluetooth (0 or 1)
- `clock_speed`: Processor clock speed in GHz
- `dual_sim`: Dual SIM support (0 or 1)
- `fc`: Front camera megapixels
- `four_g`: 4G support (0 or 1)
- `int_memory`: Internal memory in GB
- `m_dep`: Mobile depth in cm
- `mobile_wt`: Mobile weight in grams
- `n_cores`: Number of processor cores
- `pc`: Primary camera megapixels
- `px_height`: Pixel resolution height
- `px_width`: Pixel resolution width
- `ram`: RAM in MB
- `sc_h`: Screen height in cm
- `sc_w`: Screen width in cm
- `talk_time`: Talk time in hours
- `three_g`: 3G support (0 or 1)
- `touch_screen`: Touch screen (0 or 1)
- `wifi`: WiFi support (0 or 1)

**Target**:
- `price_range`: Price category (0, 1, 2, or 3)

### Step 5: Train and Deploy Model on SageMaker

#### Option A: Using Python Script (Recommended)

```python
from aws_integration import SageMakerIntegration

# Initialize
sagemaker = SageMakerIntegration(
    role='arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole',
    region='us-east-1'
)

# Upload training data
train_uri = sagemaker.upload_training_data(
    train_file='train.csv',
    bucket_name='your-bucket-name'
)

# Upload training script
script_uri = sagemaker.upload_model_script(
    script_file='sagemaker_train.py',
    bucket_name='your-bucket-name'
)

# Train model
estimator = sagemaker.train_model(
    train_data_uri=train_uri,
    script_uri=script_uri,
    instance_type='ml.m5.large',
    bucket_name='your-bucket-name'
)

# Deploy endpoint
predictor = sagemaker.deploy_endpoint(
    estimator=estimator,
    endpoint_name='mobile-price-predictor-endpoint',
    instance_type='ml.t2.medium'
)
```

#### Option B: Using AWS SageMaker Console

1. **Create Training Job**:
   - Go to SageMaker Console → Training → Training jobs
   - Click "Create training job"
   - Configure:
     - Name: `mobile-price-predictor-training`
     - Algorithm: Custom
     - Training script: Upload `sagemaker_train.py`
     - Training data: Upload your CSV file
     - Instance type: `ml.m5.large`
     - IAM role: Select your SageMaker role
   - Click "Create training job"

2. **Create Model**:
   - After training completes, go to Models
   - Create model from training job artifacts
   - Configure container settings

3. **Create Endpoint Configuration**:
   - Go to Endpoint configurations
   - Create endpoint configuration
   - Select your model
   - Instance type: `ml.t2.medium`

4. **Create Endpoint**:
   - Go to Endpoints
   - Create endpoint
   - Select your endpoint configuration
   - Name: `mobile-price-predictor-endpoint`
   - Wait for endpoint to be "InService"

#### Option C: Local Training (for testing)

```bash
python train_model.py --data train.csv --model-path model.pkl
```

**Note**: Local training is useful for testing, but you'll need to deploy to SageMaker for production use.

### Step 6: Configure Application

1. **Update `.env` file** with your endpoint name:
   ```env
   SAGEMAKER_ENDPOINT=mobile-price-predictor-endpoint
   ```

2. **Verify Endpoint Status**:
   ```bash
   python -c "from app import sagemaker_client; print(sagemaker_client.describe_endpoint(EndpointName='mobile-price-predictor-endpoint'))"
   ```

## 🎯 Usage

### Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Fill in the smartphone specifications:
   - Basic specifications (RAM, storage, processor, etc.)
   - Camera details
   - Screen specifications
   - Connectivity features
3. Click "Predict Price Range"
4. View the prediction result

### API Endpoints

#### Predict Price Range

```bash
POST /predict
Content-Type: application/json

{
  "battery_power": 2000,
  "ram": 3072,
  "int_memory": 32,
  "clock_speed": 2.0,
  "n_cores": 4,
  "pc": 12,
  "fc": 5,
  "px_height": 1960,
  "px_width": 1080,
  "sc_h": 12.5,
  "sc_w": 6.2,
  "touch_screen": 1,
  "mobile_wt": 180,
  "m_dep": 0.8,
  "talk_time": 10,
  "three_g": 1,
  "four_g": 1,
  "wifi": 1,
  "blue": 1,
  "dual_sim": 1
}
```

#### Health Check

```bash
GET /health
```

#### Endpoint Status

```bash
GET /endpoint-status
```

## 📁 Project Structure

```
mobile-price-predictor/
│
├── app.py                 # Flask application
├── aws_integration.py     # AWS SageMaker integration module
├── train_model.py         # Local training script
├── sagemaker_train.py     # SageMaker training script
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables example
├── .gitignore            # Git ignore file
├── README.md             # This file
│
├── templates/
│   └── index.html        # Frontend HTML template
│
└── static/
    ├── styles.css        # CSS styles
    └── app.js            # JavaScript for frontend
```

## 🔍 Testing

### Test Endpoint Connection

```bash
curl http://localhost:5000/health
```

### Test Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "battery_power": 2000,
    "ram": 3072,
    "int_memory": 32,
    "clock_speed": 2.0,
    "n_cores": 4,
    "pc": 12,
    "fc": 5,
    "px_height": 1960,
    "px_width": 1080,
    "sc_h": 12.5,
    "sc_w": 6.2,
    "touch_screen": 1,
    "mobile_wt": 180,
    "m_dep": 0.8,
    "talk_time": 10,
    "three_g": 1,
    "four_g": 1,
    "wifi": 1,
    "blue": 1,
    "dual_sim": 1
  }'
```

## 🐛 Troubleshooting

### Endpoint Not Found

- Verify endpoint name in `.env` file
- Check endpoint status in AWS SageMaker Console
- Ensure endpoint is "InService"

### AWS Credentials Error

- Verify AWS credentials in `.env` file
- Check IAM permissions
- Ensure credentials have SageMaker and S3 access

### Model Prediction Errors

- Verify feature order matches training data
- Check data types and ranges
- Ensure all 20 features are provided

## 💰 AWS Cost Considerations

- **SageMaker Training**: ~$0.10-0.20 per hour (ml.m5.large)
- **SageMaker Endpoint**: ~$0.05-0.10 per hour (ml.t2.medium)
- **S3 Storage**: ~$0.023 per GB/month

**Tips to reduce costs**:
- Delete endpoints when not in use
- Use smaller instance types for testing
- Clean up unused training jobs and models





