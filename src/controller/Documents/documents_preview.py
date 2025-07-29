import mimetypes, boto3
from fastapi import HTTPException
from botocore.config import Config
from supabase import create_client, Client

from src.core.config import settings

# Initialize Supabase client
supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_KEY
)

async def get_preview_url(file_name: str):
    """Get preview URL for a file"""
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
                'Key': f"{file_name}",
                'ResponseContentType': mimetypes.guess_type(file_name)[0],
                'ResponseContentDisposition': f'inline; filename="{file_name}"'
            },
            ExpiresIn=3600
        )
        return url
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting preview URL: {str(e)}"
        )   