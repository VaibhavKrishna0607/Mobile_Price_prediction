"""
AWS SageMaker Integration Module
This module handles AWS SageMaker model training, deployment, and endpoint management.
"""

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.predictor import CSVSerializer, CSVDeserializer
import pandas as pd
import os
import joblib
from datetime import datetime

class SageMakerIntegration:
    """
    Class to handle AWS SageMaker operations for mobile price prediction
    """
    
    def __init__(self, role=None, region='us-east-1'):
        """
        Initialize SageMaker integration
        
        Args:
            role: IAM role ARN for SageMaker (if None, will try to get execution role)
            region: AWS region
        """
        self.region = region
        self.session = sagemaker.Session()
        
        if role:
            self.role = role
        else:
            try:
                self.role = get_execution_role()
            except:
                raise ValueError("Could not get execution role. Please provide role ARN.")
        
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        
    def upload_training_data(self, train_file, bucket_name, s3_key_prefix='mobile-price-predictor'):
        """
        Upload training data to S3
        
        Args:
            train_file: Local path to training CSV file
            bucket_name: S3 bucket name
            s3_key_prefix: S3 key prefix for the data
            
        Returns:
            S3 URI of uploaded file
        """
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        s3_key = f"{s3_key_prefix}/data/train-{timestamp}.csv"
        
        self.s3_client.upload_file(train_file, bucket_name, s3_key)
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        
        print(f"Training data uploaded to: {s3_uri}")
        return s3_uri
    
    def upload_model_script(self, script_file, bucket_name, s3_key_prefix='mobile-price-predictor'):
        """
        Upload training script to S3
        
        Args:
            script_file: Local path to training script
            bucket_name: S3 bucket name
            s3_key_prefix: S3 key prefix
            
        Returns:
            S3 URI of uploaded script
        """
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        s3_key = f"{s3_key_prefix}/scripts/train-{timestamp}.py"
        
        self.s3_client.upload_file(script_file, bucket_name, s3_key)
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        
        print(f"Training script uploaded to: {s3_uri}")
        return s3_uri
    
    def train_model(self, train_data_uri, script_uri, instance_type='ml.m5.large', 
                   hyperparameters=None, output_path=None, bucket_name=None):
        """
        Train model using SageMaker
        
        Args:
            train_data_uri: S3 URI of training data
            script_uri: S3 URI of training script
            instance_type: SageMaker instance type
            hyperparameters: Dictionary of hyperparameters
            output_path: S3 path for model artifacts
            bucket_name: S3 bucket name for output
            
        Returns:
            Trained model estimator
        """
        if output_path is None and bucket_name:
            output_path = f"s3://{bucket_name}/mobile-price-predictor/models"
        
        # Create SKLearn estimator
        estimator = SKLearn(
            entry_point=script_uri,
            role=self.role,
            instance_type=instance_type,
            framework_version='0.24-1',
            py_version='py3',
            hyperparameters=hyperparameters or {},
            output_path=output_path
        )
        
        # Train the model
        print("Starting model training...")
        estimator.fit({'training': train_data_uri})
        
        print("Model training completed!")
        return estimator
    
    def deploy_endpoint(self, estimator, endpoint_name, instance_type='ml.t2.medium', 
                       initial_instance_count=1):
        """
        Deploy model to SageMaker endpoint
        
        Args:
            estimator: Trained model estimator
            endpoint_name: Name for the endpoint
            instance_type: Instance type for endpoint
            initial_instance_count: Number of instances
            
        Returns:
            Predictor object
        """
        print(f"Deploying endpoint: {endpoint_name}...")
        
        predictor = estimator.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer()
        )
        
        print(f"Endpoint deployed successfully: {endpoint_name}")
        return predictor
    
    def create_endpoint_config(self, model_name, endpoint_config_name, 
                              instance_type='ml.t2.medium', initial_instance_count=1):
        """
        Create endpoint configuration
        
        Args:
            model_name: Name of the SageMaker model
            endpoint_config_name: Name for endpoint configuration
            instance_type: Instance type
            initial_instance_count: Number of instances
        """
        response = self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': initial_instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1
                }
            ]
        )
        
        print(f"Endpoint configuration created: {endpoint_config_name}")
        return response
    
    def list_endpoints(self):
        """List all SageMaker endpoints"""
        response = self.sagemaker_client.list_endpoints()
        return response.get('Endpoints', [])
    
    def delete_endpoint(self, endpoint_name):
        """Delete a SageMaker endpoint"""
        print(f"Deleting endpoint: {endpoint_name}...")
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint deletion initiated: {endpoint_name}")
    
    def get_endpoint_status(self, endpoint_name):
        """Get status of an endpoint"""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            return response['EndpointStatus']
        except Exception as e:
            print(f"Error getting endpoint status: {e}")
            return None


def create_iam_role_for_sagemaker():
    """
    Helper function to create IAM role for SageMaker
    Note: This should be done through AWS Console or CLI for production
    """
    print("""
    To create IAM role for SageMaker:
    
    1. Go to AWS IAM Console
    2. Create a new role
    3. Select "SageMaker" as the service
    4. Attach policies:
       - AmazonSageMakerFullAccess
       - AmazonS3FullAccess (or restricted to your bucket)
    5. Save the role ARN
    
    Alternatively, use AWS CLI:
    aws iam create-role --role-name SageMakerExecutionRole \\
        --assume-role-policy-document file://trust-policy.json
    
    Then attach policies and save the role ARN.
    """)


if __name__ == "__main__":
    # Example usage
    print("AWS SageMaker Integration Module")
    print("This module provides functions for training and deploying models on AWS SageMaker")
    create_iam_role_for_sagemaker()

