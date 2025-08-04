import boto3
import urllib3
from src.core.config_pinecone import settings
from botocore.config import Config

# Suppress the InsecureRequestWarning for S3 connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_s3_client():
    # Configure S3 client with proper SSL settings
    config = Config(
        signature_version='s3v4',
        s3={'addressing_style': 'path'},  # Use path-style addressing
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
        config=config,
        verify=True  # Enable SSL verification
    )
    return s3_client

s3_client = create_s3_client()