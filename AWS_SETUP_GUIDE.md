# 🚀 AWS Integration Setup Guide

This guide will walk you through setting up AWS SageMaker for the Mobile Price Predictor application.

## Prerequisites

- AWS Account (create one at [aws.amazon.com](https://aws.amazon.com) if you don't have one)
- AWS CLI installed (optional but recommended)
- Basic understanding of AWS services

## Step-by-Step Setup

### Step 1: Create AWS Account and Access Keys

1. **Sign in to AWS Console**: Go to [console.aws.amazon.com](https://console.aws.amazon.com)

2. **Create IAM User** (for programmatic access):
   - Go to IAM Console → Users → Add users
   - Username: `mobile-price-predictor-user`
   - Select "Programmatic access"
   - Click Next

3. **Attach Policies**:
   - Search and attach:
     - `AmazonSageMakerFullAccess`
     - `AmazonS3FullAccess`
   - Click Next → Create user

4. **Save Credentials**:
   - **Access Key ID**: Copy this value
   - **Secret Access Key**: Copy this value (only shown once!)
   - Save these securely - you'll need them for `.env` file

### Step 2: Create IAM Role for SageMaker

1. **Go to IAM Console** → Roles → Create role

2. **Select Service**:
   - Choose "SageMaker" as the trusted entity
   - Click Next

3. **Attach Policies**:
   - Attach `AmazonSageMakerFullAccess`
   - Attach `AmazonS3FullAccess` (or create custom policy for specific bucket)
   - Click Next

4. **Name the Role**:
   - Role name: `SageMakerExecutionRole`
   - Description: "Role for SageMaker model training and deployment"
   - Click Create role

5. **Copy Role ARN**:
   - After creation, click on the role
   - Copy the "Role ARN" (format: `arn:aws:iam::ACCOUNT-ID:role/SageMakerExecutionRole`)
   - Save this for your `.env` file

### Step 3: Create S3 Bucket

1. **Go to S3 Console** → Create bucket

2. **Configure Bucket**:
   - **Bucket name**: `mobile-price-predictor-YOUR-NAME` (must be globally unique)
   - **Region**: Choose your preferred region (e.g., `us-east-1`, `ap-south-1`)
   - **Block Public Access**: Keep default settings (all blocked)
   - Click Create bucket

3. **Create Folders** (optional, for organization):
   - `data/` - for training data
   - `models/` - for model artifacts
   - `scripts/` - for training scripts

### Step 4: Prepare Training Data

Your training CSV file should have these columns:

**Required Columns:**
- `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`
- `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`
- `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`
- `talk_time`, `three_g`, `touch_screen`, `wifi`
- `price_range` (target: 0, 1, 2, or 3)

**Example data format:**
```csv
battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi,price_range
2000,1,2.0,1,5,1,32,0.8,180,4,12,1960,1080,3072,12.5,6.2,10,1,1,1,1
```

### Step 5: Upload Training Data to S3

**Option A: Using AWS Console**
1. Go to your S3 bucket
2. Click "Upload"
3. Select your training CSV file
4. Upload to `data/` folder (or root)
5. Note the S3 URI (e.g., `s3://bucket-name/data/train.csv`)

**Option B: Using AWS CLI**
```bash
aws s3 cp train.csv s3://your-bucket-name/data/train.csv
```

**Option C: Using Python Script**
```python
from aws_integration import SageMakerIntegration

sagemaker = SageMakerIntegration(
    role='arn:aws:iam::YOUR-ACCOUNT:role/SageMakerExecutionRole',
    region='us-east-1'
)

train_uri = sagemaker.upload_training_data(
    train_file='train.csv',
    bucket_name='your-bucket-name'
)
```

### Step 6: Train Model on SageMaker

#### Option A: Using Python Script (Recommended)

Create a script `deploy_model.py`:

```python
from aws_integration import SageMakerIntegration
import os

# Initialize
sagemaker = SageMakerIntegration(
    role=os.environ.get('SAGEMAKER_ROLE'),
    region=os.environ.get('AWS_REGION', 'us-east-1')
)

bucket_name = os.environ.get('S3_BUCKET_NAME')

# Upload training script
script_uri = sagemaker.upload_model_script(
    script_file='sagemaker_train.py',
    bucket_name=bucket_name
)

# Upload training data (if not already uploaded)
train_uri = sagemaker.upload_training_data(
    train_file='train.csv',
    bucket_name=bucket_name
)

# Train model
estimator = sagemaker.train_model(
    train_data_uri=train_uri,
    script_uri=script_uri,
    instance_type='ml.m5.large',  # Can use ml.t3.medium for testing
    bucket_name=bucket_name
)

# Deploy endpoint
predictor = sagemaker.deploy_endpoint(
    estimator=estimator,
    endpoint_name='mobile-price-predictor-endpoint',
    instance_type='ml.t2.medium'  # Smaller instance for cost savings
)

print("Model deployed successfully!")
print(f"Endpoint: mobile-price-predictor-endpoint")
```

Run it:
```bash
python deploy_model.py
```

#### Option B: Using SageMaker Console

1. **Create Training Job**:
   - Go to SageMaker Console → Training → Training jobs
   - Click "Create training job"
   
2. **Configure**:
   - **Name**: `mobile-price-predictor-training-YYYY-MM-DD`
   - **Algorithm**: Custom
   - **Training script**: Upload `sagemaker_train.py`
   - **Training data**: Select your S3 location
   - **Output path**: `s3://your-bucket/models/`
   - **Instance type**: `ml.m5.large` (or `ml.t3.medium` for testing)
   - **IAM role**: Select `SageMakerExecutionRole`
   
3. **Create and Wait**: Training takes 10-30 minutes

4. **Create Model**:
   - After training completes → Create model
   - Model name: `mobile-price-predictor-model`
   - Container: Use training job artifacts
   - IAM role: `SageMakerExecutionRole`

5. **Create Endpoint Configuration**:
   - Endpoint configuration name: `mobile-price-predictor-config`
   - Model: Select `mobile-price-predictor-model`
   - Instance type: `ml.t2.medium`
   - Initial instance count: 1

6. **Create Endpoint**:
   - Endpoint name: `mobile-price-predictor-endpoint`
   - Endpoint configuration: `mobile-price-predictor-config`
   - Wait for status: "InService" (5-10 minutes)

### Step 7: Configure Application

1. **Create `.env` file**:
```env
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
SAGEMAKER_ENDPOINT=mobile-price-predictor-endpoint
SAGEMAKER_ROLE=arn:aws:iam::YOUR-ACCOUNT:role/SageMakerExecutionRole
S3_BUCKET_NAME=your-bucket-name
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
PORT=5000
```

2. **Test Connection**:
```bash
python app.py
```

Visit `http://localhost:5000/health` to verify connection.

### Step 8: Test the Application

1. **Start the app**:
```bash
python app.py
```

2. **Test prediction**:
   - Open `http://localhost:5000`
   - Fill in the form
   - Click "Predict Price Range"
   - Verify prediction is returned

## Cost Optimization Tips

1. **Use Smaller Instances for Testing**:
   - Training: `ml.t3.medium` instead of `ml.m5.large`
   - Endpoint: `ml.t2.medium` is sufficient for low traffic

2. **Delete Endpoints When Not in Use**:
   ```python
   sagemaker.delete_endpoint('mobile-price-predictor-endpoint')
   ```

3. **Monitor Costs**:
   - Set up AWS Budget alerts
   - Check AWS Cost Explorer regularly

4. **Use Spot Instances for Training** (advanced):
   - Can reduce training costs by up to 70%

## Troubleshooting

### Error: "Endpoint not found"
- Verify endpoint name in `.env`
- Check SageMaker Console → Endpoints
- Ensure endpoint status is "InService"

### Error: "Access Denied"
- Verify IAM user has correct permissions
- Check IAM role has SageMaker and S3 access
- Verify AWS credentials are correct

### Error: "Training job failed"
- Check CloudWatch logs for training job
- Verify training data format is correct
- Ensure training script is uploaded correctly

### Error: "Model deployment timeout"
- Increase endpoint instance size
- Check CloudWatch logs
- Verify model artifacts are in S3

## Next Steps

1. **Monitor Performance**: Set up CloudWatch alarms
2. **Optimize Model**: Tune hyperparameters
3. **Scale Endpoint**: Adjust instance count based on traffic
4. **Set up CI/CD**: Automate model retraining

## Useful AWS CLI Commands

```bash
# List endpoints
aws sagemaker list-endpoints

# Describe endpoint
aws sagemaker describe-endpoint --endpoint-name mobile-price-predictor-endpoint

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name mobile-price-predictor-endpoint

# List training jobs
aws sagemaker list-training-jobs

# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name mobile-price-predictor-endpoint --query 'EndpointStatus'
```

## Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

---

**Need Help?** Check the main README.md or AWS documentation.

Short answer: AWS isn’t required for the project. Already trained and can predict locally. AWS just provides managed infrastructure around the same workflow.
What AWS does in this setup
S3: stores dataset, code bundle, and the trained model artifacts.
SageMaker Training: runs training script on cloud machines (instead of your laptop), handles logs, retries, and saves the output model.
SageMaker Inference (optional): hosts model behind an HTTPS endpoint you can call from a website/app.
IAM: permissions so your jobs can read S3 and pull the container image.

-When using AWS
Datasets too large for the machine, need faster/bigger GPUs/CPUs.
Need a production API endpoint with autoscaling, monitoring, and easy rollback.
Team workflows, reproducibility, audit/logging.

-When local is enough
Small dataset, training completes quickly on your laptop.
A simple local Flask app or desktop script is fine.

-current status:
Model trained locally with improved accuracy.
Data/code uploaded to S3.
Cloud training is optional; it failed due to permissions/quota, but not required to “complete” the ML part.
Easiest: run the small Flask app locally and test via browser.
Cloud API route: deploy on SageMaker endpoint (or AWS Lambda + API Gateway) so any HTML/CSS frontend can call it.