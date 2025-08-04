import os
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError

async def delete_file(request: Request):
    try:
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        user_id = body.get("userId")
        filename = body.get("filename")

        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="userId is required and cannot be empty.")

        if not filename or not filename.strip():
            raise HTTPException(status_code=400, detail="filename is required and cannot be empty.")

        s3_key = f"document-uploaded/{user_id}/{filename}"

        from src.utils.s3_client import s3_client

        # Check if S3 client is available
        if s3_client is None:
            print("WARNING: S3 client not available. Cannot delete file from S3.")
            return JSONResponse(
                status_code=200,
                content={"message": f"{filename} deletion skipped (S3 not available)"}
            )
        
        try:
            print(f"Deleting the file from S3 bucket: {s3_key}")
            s3_client.delete_object(Bucket=os.getenv("AWS_S3_BUCKET_NAME"), Key=s3_key)
            print(f"Deleted file from S3 bucket successfully.")
            return JSONResponse(
                status_code=200,
                content={"message": f"{filename} was deleted from s3 bucket successfully"}
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "NoSuchKey":
                print(f"File not found in S3: {s3_key}")
                raise HTTPException(status_code=404, detail=f"File {filename} not found in S3 bucket.")
            else:
                print(f"S3 ClientError during deletion: {error_code} - {e}")
                raise HTTPException(status_code=500, detail=f"S3 deletion error: {e}")
        except Exception as e:
            print(f"ERROR: Failed to delete file from S3: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred during file deletion: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in delete_file_route: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file deletion.")
