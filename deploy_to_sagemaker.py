"""
Deploy Mobile Price Predictor to AWS SageMaker
This script uploads data to S3, trains the model on SageMaker, and deploys an endpoint.
"""

import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
import os
import time
from datetime import datetime

# Configuration
ROLE_ARN = "arn:aws:iam::266735811100:role/SageMakerExecutionRole"
BUCKET_NAME = "mobile-price-vaibhav"
REGION = "ap-south-1"
PREFIX = "mobile-price-predictor"

# Instance types (use small instances to minimize costs)
TRAINING_INSTANCE = "ml.m5.large"
ENDPOINT_INSTANCE = "ml.t2.medium"

def upload_training_data():
    """Upload training data to S3"""
    print("\n" + "=" * 60)
    print("📤 Uploading training data to S3...")
    print("=" * 60)
    
    s3 = boto3.client('s3', region_name=REGION)
    
    # Upload the training data
    local_file = "mob_price_classification_train.csv"
    s3_key = f"{PREFIX}/data/train.csv"
    
    s3.upload_file(local_file, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"
    
    print(f"✅ Training data uploaded to: {s3_uri}")
    return s3_uri


def train_model(training_data_uri):
    """Train model on SageMaker"""
    print("\n" + "=" * 60)
    print("🚀 Starting SageMaker training job...")
    print("=" * 60)
    
    sagemaker_session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )
    
    # Create SKLearn estimator
    sklearn_estimator = SKLearn(
        entry_point="sagemaker_train.py",
        role=ROLE_ARN,
        instance_type=TRAINING_INSTANCE,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{BUCKET_NAME}/{PREFIX}/models",
        base_job_name="mobile-price-train"
    )
    
    # Start training
    print(f"Training instance: {TRAINING_INSTANCE}")
    print("This may take 5-10 minutes...")
    
    sklearn_estimator.fit({"training": training_data_uri}, wait=True)
    
    print("✅ Training completed!")
    return sklearn_estimator


def deploy_endpoint(estimator):
    """Deploy model to SageMaker endpoint"""
    print("\n" + "=" * 60)
    print("🌐 Deploying model to SageMaker endpoint...")
    print("=" * 60)
    
    endpoint_name = f"mobile-price-predictor-endpoint"
    
    print(f"Endpoint name: {endpoint_name}")
    print(f"Instance type: {ENDPOINT_INSTANCE}")
    print("This may take 5-10 minutes...")
    
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE,
        endpoint_name=endpoint_name
    )
    
    print(f"✅ Endpoint deployed: {endpoint_name}")
    return predictor, endpoint_name


def test_endpoint(endpoint_name):
    """Test the deployed endpoint"""
    print("\n" + "=" * 60)
    print("🧪 Testing endpoint...")
    print("=" * 60)
    
    runtime = boto3.client('sagemaker-runtime', region_name=REGION)
    
    # Sample test data (20 features for a mid-range phone)
    test_data = "1500,1,2.0,1,5,1,32,0.5,150,4,12,1280,720,3000,14,7,15,1,1,1"
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Accept='application/json',
        Body=test_data
    )
    
    result = response['Body'].read().decode('utf-8')
    print(f"Test input: {test_data[:50]}...")
    print(f"Prediction result: {result}")
    print("✅ Endpoint is working!")
    return result


def main():
    print("\n" + "=" * 60)
    print("🚀 Mobile Price Predictor - SageMaker Deployment")
    print("=" * 60)
    print(f"Region: {REGION}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Role: {ROLE_ARN}")
    
    try:
        # Step 1: Upload training data
        training_data_uri = upload_training_data()
        
        # Step 2: Train model
        estimator = train_model(training_data_uri)
        
        # Step 3: Deploy endpoint
        predictor, endpoint_name = deploy_endpoint(estimator)
        
        # Step 4: Test endpoint
        test_endpoint(endpoint_name)
        
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("=" * 60)
        print(f"\nEndpoint name: {endpoint_name}")
        print(f"\nTo use in your app, set environment variable:")
        print(f"  SAGEMAKER_ENDPOINT={endpoint_name}")
        print("\n⚠️  IMPORTANT: Remember to delete the endpoint when not in use")
        print("   to avoid ongoing charges. Use: python delete_endpoint.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
