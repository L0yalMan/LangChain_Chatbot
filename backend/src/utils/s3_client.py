import boto3
import os
from src.core.config import settings
from botocore.config import Config

def create_s3_client():
    """Create S3 client with proper error handling."""
    try:
        # Get AWS credentials and region
        aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        aws_region = settings.AWS_REGION
        
        # Validate required parameters
        if not aws_access_key_id or not aws_secret_access_key:
            print("WARNING: AWS credentials not configured. S3 functionality will be disabled.")
            return None
            
        if not aws_region:
            print("WARNING: AWS region not configured. Using default region 'us-east-1'")
            aws_region = 'us-east-1'
        
        # Validate region format (should not contain double dots or invalid characters)
        if '..' in aws_region or not aws_region.replace('-', '').replace('_', '').isalnum():
            print(f"WARNING: Invalid AWS region format: {aws_region}. Using default region 'us-east-1'")
            aws_region = 'us-east-1'
        
        print(f"DEBUG: Creating S3 client with region: {aws_region}")
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
            config=Config(signature_version='s3v4')
        )
        
        print("DEBUG: S3 client created successfully")
        return s3_client
        
    except Exception as e:
        print(f"ERROR: Failed to create S3 client: {e}")
        print("S3 functionality will be disabled.")
        return None

# Create S3 client with error handling
s3_client = create_s3_client()

