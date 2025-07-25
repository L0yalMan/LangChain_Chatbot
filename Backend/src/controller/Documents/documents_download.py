import mimetypes, boto3
from fastapi import HTTPException
from botocore.config import Config
from src.core.config import settings
# from src.utils.supabase_client import supabase

async def get_download_url(file_name: str):
    """Get download URL for a file"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            config=Config(signature_version='s3v4')
        )
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.AWS_S3_BUCKET_NAME,
                'Key': f"{file_name}"
            },
            ExpiresIn=3600
        )
        return url
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting download URL: {str(e)}"
        )