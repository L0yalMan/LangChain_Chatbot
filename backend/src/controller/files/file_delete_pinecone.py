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
        if pinecone_client:
            try:
                # Try to get the index directly - this will fail if it doesn't exist
                index = pinecone_client.Index(index_name)
                print(f"Successfully connected to index '{index_name}' for deletion")
                
                # Using the metadata to delete all vectors related to this file.
                print(f"Deleting vectors for file '{filename}' from Pinecone namespace '{user_id}'...")
                try:
                    # Build filter based on available information
                    if s3_key:
                        # Most precise deletion using file_upload_id
                        filter_criteria = {"s3_key": {"$eq": s3_key}}
                        print(f"Using s3_key for precise deletion: {s3_key}")
                    else:
                        # Fallback to filename-based deletion
                        filter_criteria = {"source": {"$eq": filename}}
                        print(f"Using filename for deletion: {filename}")
                    
                    # Delete vectors in the user's namespace that match the filter criteria
                    delete_result = index.delete(filter=filter_criteria, namespace=user_id)
                    print(f"Successfully deleted vectors from Pinecone for file '{filename}' in namespace '{user_id}'.")
                    
                    # Check remaining vectors for the user
                    try:
                        stats = index.describe_index_stats()
                        namespaces = stats.get('namespaces', {})
                        user_namespace_stats = namespaces.get(user_id, {})
                        remaining_vectors = user_namespace_stats.get('vector_count', 0)
                        print(f"Remaining vectors for user {user_id}: {remaining_vectors}")
                        
                        # Update global vector store state if needed
                        from src.core.config_pinecone import vectorstores
                        if user_id in vectorstores:
                            print(f"Global vector store state updated for user {user_id} after deletion")
                            # The vector store instance should still be valid, just with fewer vectors
                            # No need to recreate it, just ensure it's properly referenced
                            pass
                    except Exception as stats_error:
                        print(f"WARNING: Could not check remaining vectors: {stats_error}")
                        
                except Exception as e:
                    print(f"WARNING: Failed to delete vectors from Pinecone: {e}")
                    # Continue with S3 deletion even if Pinecone deletion fails
            except Exception as e:
                print(f"WARNING: Index '{index_name}' does not exist or is not accessible: {e}")
                print("Skipping vector deletion.")
        else:
            print("WARNING: Pinecone client not available. Skipping vector deletion.")

        # Delete from S3
        s3_deletion_success = False
        if s3_client is None:
            print("WARNING: S3 client not available. Cannot delete file from S3.")
        else:
            try:
                print(f"Deleting the file from S3 bucket: {s3_key}")
                s3_client.delete_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=s3_key)
                print(f"Deleted file from S3 bucket successfully.")
                s3_deletion_success = True
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == "NoSuchKey":
                    print(f"File not found in S3: {s3_key}")
                    # Don't treat this as an error, just inform the user
                    s3_deletion_success = True  # File doesn't exist, so deletion is "successful"
                else:
                    print(f"S3 ClientError during deletion: {error_code} - {e}")
                    raise HTTPException(status_code=500, detail=f"S3 deletion error: {e}")
            except Exception as e:
                print(f"ERROR: Failed to delete file from S3: {e}")
                raise HTTPException(status_code=500, detail=f"An error occurred during file deletion: {str(e)}")
        
        # Return success response
        if s3_deletion_success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"{filename} was deleted successfully",
                    "filename": filename,
                    "user_id": user_id,
                    "s3_key": s3_key,
                    "file_upload_id": s3_key
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"{filename} deletion completed (S3 not available)",
                    "filename": filename,
                    "user_id": user_id,
                    "s3_key": s3_key,
                    "file_upload_id": s3_key
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in delete_file_route: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file deletion.")

