"""
AWS Connection Test Script
Run this to verify your AWS configuration is working correctly
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
import sys

def test_aws_credentials():
    """Test if AWS credentials are valid"""
    print("=" * 60)
    print("🔐 Testing AWS Credentials...")
    print("=" * 60)
    
    try:
        # Try to get caller identity (checks if credentials work)
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        print(f"✅ AWS Credentials: VALID")
        print(f"   Account ID: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        return True
    except NoCredentialsError:
        print("❌ AWS Credentials: NOT FOUND")
        print("   Please configure AWS credentials using 'aws configure' or set environment variables")
        return False
    except ClientError as e:
        print(f"❌ AWS Credentials: INVALID - {e}")
        return False


def test_s3_access():
    """Test S3 access"""
    print("\n" + "=" * 60)
    print("📦 Testing S3 Access...")
    print("=" * 60)
    
    try:
        s3 = boto3.client('s3')
        response = s3.list_buckets()
        buckets = response.get('Buckets', [])
        
        print(f"✅ S3 Access: OK")
        print(f"   Found {len(buckets)} bucket(s):")
        for bucket in buckets[:5]:  # Show first 5 buckets
            print(f"      - {bucket['Name']}")
        if len(buckets) > 5:
            print(f"      ... and {len(buckets) - 5} more")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print("⚠️  S3 Access: LIMITED (no ListBuckets permission)")
            print("   You may still be able to access specific buckets")
            return True
        print(f"❌ S3 Access: FAILED - {e}")
        return False
    except Exception as e:
        print(f"❌ S3 Access: FAILED - {e}")
        return False


def test_sagemaker_access():
    """Test SageMaker access"""
    print("\n" + "=" * 60)
    print("🤖 Testing SageMaker Access...")
    print("=" * 60)
    
    try:
        sagemaker = boto3.client('sagemaker')
        
        # List endpoints (lightweight call)
        response = sagemaker.list_endpoints(MaxResults=5)
        endpoints = response.get('Endpoints', [])
        
        print(f"✅ SageMaker Access: OK")
        print(f"   Found {len(endpoints)} endpoint(s):")
        for endpoint in endpoints:
            print(f"      - {endpoint['EndpointName']} ({endpoint['EndpointStatus']})")
        if len(endpoints) == 0:
            print("      (No endpoints deployed yet)")
        
        # Check for training jobs
        jobs_response = sagemaker.list_training_jobs(MaxResults=5)
        jobs = jobs_response.get('TrainingJobSummaries', [])
        print(f"   Found {len(jobs)} training job(s):")
        for job in jobs:
            print(f"      - {job['TrainingJobName']} ({job['TrainingJobStatus']})")
        if len(jobs) == 0:
            print("      (No training jobs yet)")
            
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print("❌ SageMaker Access: DENIED")
            print("   Your IAM user needs SageMaker permissions")
            print("   Attach 'AmazonSageMakerFullAccess' policy to your IAM user")
        else:
            print(f"❌ SageMaker Access: FAILED - {e}")
        return False
    except Exception as e:
        print(f"❌ SageMaker Access: FAILED - {e}")
        return False


def test_iam_role():
    """Test if SageMaker execution role exists"""
    print("\n" + "=" * 60)
    print("👤 Testing IAM Role for SageMaker...")
    print("=" * 60)
    
    try:
        iam = boto3.client('iam')
        role_name = 'SageMakerExecutionRole'
        
        try:
            response = iam.get_role(RoleName=role_name)
            role = response['Role']
            print(f"✅ IAM Role '{role_name}': FOUND")
            print(f"   Role ARN: {role['Arn']}")
            print(f"   Use this ARN in your .env file as SAGEMAKER_ROLE")
            return role['Arn']
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                print(f"⚠️  IAM Role '{role_name}': NOT FOUND")
                print("   You need to create a SageMaker execution role")
                print("   Follow the AWS_SETUP_GUIDE.md Step 2")
            else:
                print(f"❌ IAM Role check failed: {e}")
            return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDenied':
            print("⚠️  IAM Access: Cannot check roles (missing IAM permissions)")
            print("   This is optional - you can still use SageMaker if you have the role ARN")
        return None


def test_sagemaker_endpoint_invoke(endpoint_name='mobile-price-predictor-endpoint'):
    """Test invoking a SageMaker endpoint"""
    print("\n" + "=" * 60)
    print(f"🔮 Testing SageMaker Endpoint Invocation...")
    print("=" * 60)
    
    try:
        runtime = boto3.client('sagemaker-runtime')
        
        # Sample test data (20 features for mobile phone)
        test_data = "2000,1,2.0,1,5,1,32,0.8,180,4,12,1960,1080,3072,12.5,6.2,10,1,1,1"
        
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_data
        )
        
        result = response['Body'].read().decode('utf-8')
        print(f"✅ Endpoint '{endpoint_name}': WORKING")
        print(f"   Test prediction result: {result}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationError' and 'Could not find endpoint' in str(e):
            print(f"⚠️  Endpoint '{endpoint_name}': NOT FOUND")
            print("   This is expected if you haven't deployed a model yet")
            print("   Deploy a model first using aws_integration.py")
        else:
            print(f"❌ Endpoint invocation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Endpoint invocation failed: {e}")
        return False


def get_region():
    """Get the current AWS region"""
    session = boto3.session.Session()
    return session.region_name


def main():
    print("\n" + "🚀 AWS Integration Test for Mobile Price Predictor 🚀")
    print("=" * 60)
    print(f"Region: {get_region()}")
    print("=" * 60)
    
    results = {
        'credentials': test_aws_credentials(),
        's3': test_s3_access(),
        'sagemaker': test_sagemaker_access(),
        'iam_role': test_iam_role(),
        'endpoint': test_sagemaker_endpoint_invoke()
    }
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL/MISSING"
        print(f"   {test.upper()}: {status}")
    
    print("\n" + "=" * 60)
    print("📝 NEXT STEPS")
    print("=" * 60)
    
    if not results['credentials']:
        print("1. Configure AWS credentials:")
        print("   Run: aws configure")
        print("   Or create a .env file with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    elif not results['sagemaker']:
        print("1. Grant SageMaker permissions to your IAM user:")
        print("   Attach 'AmazonSageMakerFullAccess' policy")
    elif not results['iam_role']:
        print("1. Create SageMaker execution role:")
        print("   Follow AWS_SETUP_GUIDE.md Step 2")
    elif not results['endpoint']:
        print("1. Deploy a model to create an endpoint:")
        print("   Use the aws_integration.py module to train and deploy")
        print("   Or run the app with USE_MOCK_PREDICTIONS=true for testing")
    else:
        print("🎉 All AWS services are configured correctly!")
        print("   You can now use the full AWS integration")
    
    print()
    return all(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
