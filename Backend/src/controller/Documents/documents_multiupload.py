from botocore.exceptions import ClientError
from src.core.config import settings
from src.scalars.Documents.documents_request import CompleteMultipartUploadRequest
from src.utils.s3_client import s3_client
from src.utils.supabase_client import supabase

async def complete_multipart_upload(request: CompleteMultipartUploadRequest):
    """Complete a multipart upload"""
    print(request)
    try:
        # First, verify that the multipart upload exists
        try:
            s3_client.list_parts(
                Bucket=settings.AWS_S3_BUCKET_NAME,
                Key=request.file_name,
                UploadId=request.upload_id
            )
        except ClientError as e:
            raise e

        # Convert parts to the format S3 expects
        formatted_parts = []
        for part in request.parts:
            formatted_parts.append({
                'PartNumber': part.PartNumber,
                'ETag': part.ETag
            })

        # Sort parts by part number
        formatted_parts.sort(key=lambda x: x['PartNumber'])

        response = s3_client.complete_multipart_upload(
            Bucket=settings.AWS_S3_BUCKET_NAME,
            Key=request.file_name,
            UploadId=request.upload_id,
            MultipartUpload={'Parts': formatted_parts}
        )
        
        return {
            'status': 'success',
            'message': 'Multipart upload completed successfully',
            'location': response.get('Location', ''),
            'bucket': response.get('Bucket', ''),
            'key': response.get('Key', '')
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"AWS Error: {error_code} - {error_message}")
        print(e)
        raise e

    except Exception as e:
        print(f"General Error: {str(e)}")
        raise e
