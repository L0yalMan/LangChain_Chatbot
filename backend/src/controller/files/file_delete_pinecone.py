from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError

from src.core.config_pinecone import get_pinecone_client, settings
from src.core.retriver_pinecone import index
from src.utils.dependencies import TokenData
from src.utils.s3_client import s3_client

async def delete_file(request: Request, current_user: TokenData):
    """
    Deletes a file from S3 and its corresponding vectors from the user's Pinecone namespace.
    """
    user_id = current_user.user_id

    try:
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        filename = body.get("filename")

        if not filename or not filename.strip():
            raise HTTPException(status_code=400, detail="filename is required and cannot be empty.")

        s3_key = f"document-uploaded/{user_id}/{filename}"
        index_name = settings.PINECONE_INDEX_NAME
        
        # Delete from Pinecone
        pinecone_client = get_pinecone_client()
        existing_indexes = pinecone_client.list_indexes()
        if pinecone_client and index_name in existing_indexes:
            index = pinecone_client.Index(index_name)
            
            # Using the `source` metadata to delete all vectors related to this file.
            print(f"Deleting vectors for file '{filename}' from Pinecone namespace '{user_id}'...")
            try:
                # Deleting the entire index would remove all data for all users, not just a specific user.
                # To delete only the vectors for a specific file (for a specific user/namespace), use a filter as below.
                # This deletes all vectors in the user's namespace that have the "source" metadata matching the s3_key.
                index.delete(filter={"source": {"$eq": s3_key}}, namespace=user_id)
                print(f"Successfully deleted vectors from Pinecone for file '{filename}' in namespace '{user_id}'.")
            except Exception as e:
                print(f"WARNING: Failed to delete vectors from Pinecone: {e}")
        else:
            print("WARNING: Pinecone not initialized or index not found. Skipping vector deletion.")

        # Delete from S3
        if s3_client is None:
            print("WARNING: S3 client not available. Cannot delete file from S3.")
            return JSONResponse(
                status_code=200,
                content={"message": f"{filename} deletion skipped (S3 not available)"}
            )
        
        try:
            print(f"Deleting the file from S3 bucket: {s3_key}")
            s3_client.delete_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=s3_key)
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

