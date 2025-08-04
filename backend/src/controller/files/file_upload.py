import os
import shutil
from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from langchain_chroma import Chroma

from src.core.config import CHROMA_PATH, update_vector_store_globals
from src.core.retriever import create_advanced_retriever
from src.utils.document_processing import load_document, split_documents
from src.utils.dependencies import TokenData

async def upload_file(current_user: TokenData, file: UploadFile = File(...)):
    print(f"Received file: {file.filename if file else 'None'}")
    print(f"Received user_id: {current_user.user_id}")
    print(f"File object type: {type(file)}")
    if file:
        print(f"File filename: {file.filename}")
        print(f"File content_type: {file.content_type}")
        print(f"File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
    """
    Handles file uploads. It appends the new document to the existing ChromaDB vector store if it exists,
    or creates a new one if not. Also uploads the file to S3.
    """
    from src.core.config import embeddings, vectorstore
    from src.utils.s3_client import s3_client

    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided or invalid file.")

        file_extension = os.path.splitext(file.filename)[1]
        if file_extension not in [".pdf", ".csv", ".docx", ".doc"]:
            raise HTTPException(status_code=400, detail="Only PDF, CSV, DOCX, and DOC files are supported.")

        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings not initialized. Please check your Google API key.")

        temp_dir = "temp_docs"
        file_path = os.path.join(temp_dir, file.filename)
        s3_key = f"document-uploaded/{current_user.user_id}/{file.filename}"

        try:
            os.makedirs(temp_dir, exist_ok=True)
            print("Creating temporary directory and saving uploaded file...")

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Upload to S3 if client is available
            if s3_client is not None:
                with open(file_path, "rb") as data:
                    print(f"Uploading file to S3... {file.filename}")
                    s3_client.upload_fileobj(data, os.getenv("AWS_S3_BUCKET_NAME"), s3_key)
                print(f"File '{file.filename}' uploaded to S3 at '{s3_key}'.")
            else:
                print("WARNING: S3 client not available. File will not be uploaded to S3.")
                s3_key = "local_storage_only"

            docs = load_document(file_path, file_extension)
            if not docs:
                raise HTTPException(status_code=400, detail="No content could be extracted from the uploaded file.")

            total_content_length = sum(len(doc.page_content) for doc in docs)
            splits = split_documents(docs, total_content_length)

            if not splits:
                raise HTTPException(status_code=400, detail="No text chunks could be created from the document.")

            print(f"--- CHUNKS CREATED FROM UPLOADED FILE: {file.filename} ---")
            print(f"Total chunks created: {len(splits)}")
            for i, chunk in enumerate(splits):
                try:
                    print(f"\n--- CHUNK {i+1}/{len(splits)} ---")
                    print(f"Content: {chunk.page_content[:200]}...")
                    print(f"Metadata: {chunk.metadata}")
                    print("-" * 50)
                except Exception as e:
                    print(f"ERROR: Failed to print chunk {i+1}: {e}")

            if vectorstore:
                print("--- APPENDING TO EXISTING VECTOR STORE ---")
                vectorstore.add_documents(splits)
            else:
                print("--- CREATING NEW VECTOR STORE ---")
                new_vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=CHROMA_PATH
                )
                update_vector_store_globals(new_vectorstore, None)  # Will update retriever later

            # Recreate retriever with updated vector store
            current_vectorstore = vectorstore  # Get the current vectorstore (might have been updated)
            new_retriever = create_advanced_retriever(current_vectorstore)
            if new_retriever is None:
                print("WARNING: Failed to create advanced retriever after upload")
            else:
                update_vector_store_globals(current_vectorstore, new_retriever)

            try:
                collection_count = vectorstore._collection.count()
                print(f"--- VECTOR STORE UPDATED ---")
                print(f"Total chunks in vector store after upload: {collection_count}")
            except Exception as e:
                print(f"Could not get updated collection count: {e}")

            print(f"File '{file.filename}' processed. Vector store is ready in '{CHROMA_PATH}'.")

            # Determine success message based on S3 availability
            if s3_client is not None:
                message = f"File '{file.filename}' uploaded, processed, and stored in S3 successfully."
            else:
                message = f"File '{file.filename}' processed and stored locally (S3 not available)."
            
            return JSONResponse(
                status_code=200,
                content={"message": message, "key": f"{s3_key}"}
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
