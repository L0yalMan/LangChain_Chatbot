import mimetypes
from typing import List

from src.core.config import settings
from src.scalars.Documents.documents_response import FileUpResponse
from src.scalars.Documents.documents_request import FileUpRequest
from src.utils.s3_client import s3_client
# from src.utils.supabase_client import supabase


async def upload_files(request: List[FileUpRequest]):
    upload_results = []
    for file_request in request:
        try:
            user_id = file_request.user_id
            original_key = file_request.file_name
            key = f"document-uploaded/{user_id}/{original_key}"

            s3_path = f"s3://{settings.AWS_S3_BUCKET_NAME}/{key}"

            if file_request.part_num == 1:
                url = []
                try:
                    url.append(s3_client.generate_presigned_url(
                        'put_object',
                        Params={
                            'Bucket': settings.AWS_S3_BUCKET_NAME,
                            'Key': key,
                            'ContentType': mimetypes.guess_type(key)[0]
                        },
                        ExpiresIn=3600
                    ))

                    upload_results.append(FileUpResponse(
                        file_name=key,
                        presigned_url=url,
                        s3_path=s3_path,
                        message="Presigned URL generated successfully"
                    ))
                except Exception as e:
                    upload_results.append(FileUpResponse(
                        file_name=key,
                        message=f"Error generating presigned URL: {str(e)}"
                    ))

            else:
                try:
                    # Multipart file upload
                    upload = s3_client.create_multipart_upload(
                        Bucket=settings.AWS_S3_BUCKET_NAME,
                        Key=key,
                    )
                    upload_id = upload["UploadId"]
                    urls = []
                    part_num = []
                    path = f"s3://{settings.AWS_S3_BUCKET_NAME}/{key}"
                    for chunk_num in range(file_request.part_num):
                        chunk_url = s3_client.generate_presigned_url(
                            'upload_part',
                            Params={
                                'Bucket': settings.AWS_S3_BUCKET_NAME,
                                'Key': key,
                                'UploadId': upload_id,
                                'PartNumber': chunk_num + 1
                            },
                            ExpiresIn=10000,
                            HttpMethod='PUT'
                        )
                        urls.append(chunk_url)
                        part_num.append(chunk_num + 1)

                    upload_results.append(FileUpResponse(
                        file_name=key,
                        upload_id=upload_id,
                        presigned_url=urls,
                        s3_path=path,
                        part_num=part_num,
                        message=f"Generated {file_request.part_num} presigned URLs for multipart upload"
                    ))
                except Exception as e:
                    upload_results.append(FileUpResponse(
                        file_name=key,
                        message=f"Error generating presigned URL: {str(e)}"
                    ))

        except Exception as e:
            upload_results.append(FileUpResponse(
                file_name=key,
                message=f"Error generating presigned URL: {str(e)}"
            ))

    return upload_results