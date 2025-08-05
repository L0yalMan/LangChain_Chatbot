import os
import shutil
from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from langchain_pinecone import PineconeVectorStore

from src.core.config_pinecone import update_vector_store_globals
from src.core.retriver_pinecone import create_advanced_retriever, index
from src.utils.document_processing import load_document, split_documents
from src.utils.dependencies import TokenData

async def upload_file(current_user: TokenData, file: UploadFile = File(...)):
    """
    Handles file uploads. It appends the new document to the user's Pinecone vector store
    and also uploads the file to S3.
    """
    user_id = current_user.user_id
    from src.core.config_pinecone import embeddings, vectorstores
    from src.utils.s3_client import s3_client

    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided.")
        
        file_extension = os.path.splitext(file.filename)[1]
        if file_extension not in [".pdf", ".csv", ".docx", ".doc"]:
            raise HTTPException(status_code=400, detail="Only PDF, CSV, DOCX, and DOC files are supported.")

        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings model not initialized.")

        temp_dir = "temp_docs"
        file_path = os.path.join(temp_dir, file.filename)
        s3_key = f"document-uploaded/{current_user.user_id}/{file.filename}"

        try:
            os.makedirs(temp_dir, exist_ok=True)
            print("Creating temporary directory and saving uploaded file...")
            
            from src.core.config_pinecone import settings
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)


            docs = load_document(file_path, file_extension)
            if not docs:
                raise HTTPException(status_code=400, detail="No content could be extracted from the uploaded file.")

            total_content_length = sum(len(doc.page_content) for doc in docs)
            splits = split_documents(docs, total_content_length)

            if not splits:
                raise HTTPException(status_code=400, detail="No text chunks could be created from the document.")
            
            for i, chunk in enumerate(splits):
                # Add comprehensive metadata for per-file tracking and deletion
                chunk.metadata.update({
                    'source': file.filename,  # Original filename
                    'user_id': user_id,       # User isolation
                    'chunk_index': i,
                    'total_chunks': len(splits),
                    'file_extension': file_extension,
                    's3_key': s3_key,  # S3 storage key for file deletion
                    'file_size': os.path.getsize(file_path)
                })
            
            print(f"--- CHUNKS CREATED FROM UPLOADED FILE: {file.filename} ---")
            print(f"Total chunks created: {len(splits)}")
            print(f"S3 Key: {s3_key}")
            
            for i, chunk in enumerate(splits):
                try:
                    print(f"\n--- CHUNK {i+1}/{len(splits)} ---")
                    print(f"Content: {chunk.page_content[:200]}...")
                    print(f"Metadata: {chunk.metadata}")
                    print("-" * 50)
                except Exception as e:
                    print(f"ERROR: Failed to print chunk {i+1}: {e}")

            # Get Pinecone client and index name early for all code paths
            from src.core.retriver_pinecone import load_existing_vector_store
            from src.core.config_pinecone import get_pinecone_client
            
            pinecone_client = get_pinecone_client()
            index_name = settings.PINECONE_INDEX_NAME
            
            vectorstore = vectorstores.get(user_id)
            if vectorstore:
                print("--- APPENDING TO EXISTING VECTOR STORE ---")
                vectorstore.add_documents(splits)
                # Update the global vectorstore reference
                update_vector_store_globals(user_id, vectorstore, None)
            else:
                print("--- LOADING OR CREATING VECTOR STORE ---")
                
                # Try to load existing vector store first
                existing_vectorstore = load_existing_vector_store(user_id, index_name, embeddings)
                
                if existing_vectorstore:
                    print("--- LOADED EXISTING VECTOR STORE ---")
                    vectorstore = existing_vectorstore
                    vectorstore.add_documents(splits)
                else:
                    print("--- CREATING NEW VECTOR STORE ---")
                    # Ensure the index exists before creating vector store
                    from src.core.retriver_pinecone import ensure_index_exists
                    from src.core.config_pinecone import settings
                    
                    if ensure_index_exists(index_name, pinecone_client, settings):
                        new_vectorstore = PineconeVectorStore(
                            index=pinecone_client.Index(index_name),
                            embedding=embeddings,
                            text_key="text",
                            namespace=user_id
                        )
                        vectorstore = new_vectorstore
                    else:
                        raise HTTPException(status_code=500, detail="Failed to create or access Pinecone index")
                try:
                    stats = index.describe_index_stats()
                    namespaces = stats.get('namespaces', {})
                    user_namespace_stats = namespaces.get(user_id, {})
                    remaining_vectors = user_namespace_stats.get('vector_count', 0)
                    print(f"Remaining vectors for user {user_id}: {remaining_vectors}")
                except Exception as e:
                    print(f"ERROR: Failed to Remaining vectors for user {user_id}: {e}")
                update_vector_store_globals(user_id, vectorstore, None)


            # Re-initialize the retriever for the user
            try:
                current_vectorstore = vectorstore
                new_retriever = create_advanced_retriever(current_vectorstore)
                if new_retriever is None:
                    print("WARNING: Failed to create advanced retriever after file upload")
                    # Update with None retriever
                    update_vector_store_globals(user_id, current_vectorstore, None)
                else:
                    # Update with new retriever
                    update_vector_store_globals(user_id, current_vectorstore, new_retriever)
            except Exception as retriever_error:
                print(f"ERROR: Failed to create retriever after file upload: {retriever_error}")
                # Update with None retriever as fallback
                update_vector_store_globals(user_id, vectorstore, None)

            try:
                # Get collection count using the new Pinecone API
                if pinecone_client and index_name:
                    stats = pinecone_client.Index(index_name).describe_index_stats()
                    namespaces = stats.get('namespaces', {})
                    user_namespace_stats = namespaces.get(user_id, {})
                    collection_count = user_namespace_stats.get('vector_count', 0)
                    print(f"--- VECTOR STORE UPDATED ---")
                    print(f"Total chunks in vector store after upload: {collection_count}")
                else:
                    print("WARNING: Pinecone client or index name not available for collection count")
            except Exception as e:
                print(f"Could not get updated collection count: {e}")

            print(f"File '{file.filename}' processed. Vector store is ready in Pinecone.")

            # Upload to S3 if client is available
            if s3_client is not None:
                with open(file_path, "rb") as data:
                    print(f"Uploading file to S3... {file.filename}")
                    s3_client.upload_fileobj(data, settings.AWS_S3_BUCKET_NAME , s3_key)
                print(f"File '{file.filename}' uploaded to S3 at '{s3_key}'.")
            else:
                print("WARNING: S3 client not available. File will not be uploaded to S3.")
                s3_key = "local_storage_only"

            # Determine success message based on S3 availability
            if s3_client is not None:
                message = f"File '{file.filename}' uploaded, processed, and stored in S3 successfully."
            else:
                message = f"File '{file.filename}' processed and stored locally (S3 not available)."

            return JSONResponse(
                status_code=200,
                content={
                    "message": message, 
                    "key": s3_key,
                    "chunks_processed": len(splits),
                    "file_size": os.path.getsize(file_path),
                    "user_id": user_id,
                    "filename": file.filename
                }
            )

        except HTTPException:
            raise
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            print(f"S3 ClientError during upload: {error_code} - {e}")
            raise HTTPException(status_code=500, detail=f"S3 upload error (ClientError): {e}")
        except Exception as e:
            print(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        finally:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                if not file.file.closed:
                    file.file.close()
            except Exception as e:
                print(f"ERROR: Failed to cleanup temporary files: {e}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in upload_file_route: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file upload.") 