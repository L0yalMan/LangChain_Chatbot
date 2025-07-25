import boto3
from src.core.config import settings
from botocore.config import Config

def  create_s3_client():
  s3_client = boto3.client(
      's3',
      aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
      aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
      region_name=settings.AWS_REGION,
      config=Config(signature_version='s3v4')
  )
  return s3_client

s3_client = create_s3_client()

