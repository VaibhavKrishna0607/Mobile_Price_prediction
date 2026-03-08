"""
Delete SageMaker Endpoint
Run this script to delete the endpoint and avoid ongoing charges.
"""

import boto3

REGION = "ap-south-1"
ENDPOINT_NAME = "mobile-price-predictor-endpoint"


def delete_endpoint():
    """Delete the SageMaker endpoint and associated resources"""
    print(f"Deleting endpoint: {ENDPOINT_NAME}")
    
    sagemaker = boto3.client('sagemaker', region_name=REGION)
    
    try:
        # Get endpoint config name
        endpoint_info = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        endpoint_config_name = endpoint_info['EndpointConfigName']
        
        # Delete endpoint
        print("Deleting endpoint...")
        sagemaker.delete_endpoint(EndpointName=ENDPOINT_NAME)
        
        # Wait for deletion
        print("Waiting for endpoint to be deleted...")
        waiter = sagemaker.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=ENDPOINT_NAME)
        
        # Delete endpoint config
        print(f"Deleting endpoint config: {endpoint_config_name}")
        sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        
        print("✅ Endpoint and config deleted successfully!")
        
    except sagemaker.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            print("Endpoint not found - already deleted or never created.")
        else:
            raise


if __name__ == "__main__":
    delete_endpoint()
