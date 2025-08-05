from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_pinecone import PineconeVectorStore

from src.core.config_pinecone import update_vector_store_globals, settings, get_pinecone_client
from src.core.retriver_pinecone import create_advanced_retriever, index
from src.utils.document_processing import load_website_content, split_documents
from src.utils.dependencies import TokenData

async def ingest_website_link(request: Request, current_user: TokenData):
    """
    Ingests content from a given website URL into the RAG system.
    It scrapes the content and adds it to the Pinecone vector store for the user.
    """
    user_id = current_user.user_id
    from src.core.config_pinecone import embeddings, vectorstores

    try:
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        url = body.get("url")

        if not url or not url.strip():
            raise HTTPException(status_code=400, detail="Website URL not provided or empty.")

        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

        if not embeddings:
            raise HTTPException(status_code=500, detail="Embeddings model not initialized.")

        # Load and split documents
        try:
            docs = load_website_content(url)
            if not docs:
                raise HTTPException(status_code=400, detail="No content could be extracted from the provided URL.")
            print(f"Successfully loaded {len(docs)} documents from {url}")
        except Exception as e:
            print(f"ERROR: Failed to load content from URL: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load content from URL: {str(e)}. Please check the URL and try again."
            )

        total_content_length = sum(len(doc.page_content) for doc in docs)
        split_docs = split_documents(docs, total_content_length)

        if not split_docs:
            raise HTTPException(status_code=400, detail="No text chunks could be created from the website content.")

        print(f"--- CHUNKS CREATED FROM WEBSITE: {url} ---")
        print(f"Total chunks created: {len(split_docs)}")
        for i, chunk in enumerate(split_docs):
            try:
                print(f"\n--- CHUNK {i+1}/{len(split_docs)} ---")
                print(f"Content: {chunk.page_content[:200]}...")
                print(f"Metadata: {chunk.metadata}")
                print("-" * 50)
            except Exception as e:
                print(f"ERROR: Failed to print chunk {i+1}: {e}")
                
        # Add metadata for deletion
        for doc in split_docs:
            doc.metadata['source'] = url
            doc.metadata['user_id'] = user_id

        # Get Pinecone client and index name early for all code paths
        from src.core.retriver_pinecone import load_existing_vector_store
        from src.core.config_pinecone import get_pinecone_client
        
        pinecone_client = get_pinecone_client()
        index_name = settings.PINECONE_INDEX_NAME
        
        vectorstore = vectorstores.get(user_id)
        if vectorstore:
            print("--- APPENDING WEBSITE CONTENT TO EXISTING VECTOR STORE ---")
            vectorstore.add_documents(split_docs)
            # Update the global vectorstore reference
            update_vector_store_globals(user_id, vectorstore, None)
        else:
            print("--- LOADING OR CREATING VECTOR STORE FROM WEBSITE CONTENT ---")
            
            # Try to load existing vector store first
            existing_vectorstore = load_existing_vector_store(user_id, index_name, embeddings)
            
            if existing_vectorstore:
                print("--- LOADED EXISTING VECTOR STORE ---")
                vectorstore = existing_vectorstore
                vectorstore.add_documents(split_docs)
            else:
                print("--- CREATING NEW VECTOR STORE ---")
                # Ensure the index exists before creating vector store
                from src.core.retriver_pinecone import ensure_index_exists
                
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
            
            update_vector_store_globals(user_id, vectorstore, None)

        # Re-initialize the retriever for the user
        try:
            current_vectorstore = vectorstore
            new_retriever = create_advanced_retriever(current_vectorstore)
            if new_retriever is None:
                print("WARNING: Failed to create advanced retriever after website ingestion")
                # Update with None retriever
                update_vector_store_globals(user_id, current_vectorstore, None)
            else:
                # Update with new retriever
                update_vector_store_globals(user_id, current_vectorstore, new_retriever)
        except Exception as retriever_error:
            print(f"ERROR: Failed to create retriever after website ingestion: {retriever_error}")
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
                print(f"Total chunks in vector store after website ingestion: {collection_count}")
            else:
                print("WARNING: Pinecone client or index name not available for collection count")
        except Exception as e:
            print(f"Could not get updated collection count: {e}")

        print(f"Website content from '{url}' processed and added to vector store.")

        return JSONResponse(
            status_code=200,
            content={"message": f"Content from '{url}' ingested successfully."}
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in website ingestion: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during website ingestion."
        )

async def delete_website_link(request: Request, current_user: TokenData):
    user_id = current_user.user_id
    try:
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        url = body.get("url")
        index_name = settings.PINECONE_INDEX_NAME
        
        # Delete from Pinecone
        pinecone_client = get_pinecone_client()
        if pinecone_client:
            try:
                # Try to get the index directly - this will fail if it doesn't exist
                index = pinecone_client.Index(index_name)
                print(f"Successfully connected to index '{index_name}' for deletion")
                
                # Using the metadata to delete all vectors related to this file.
                try:
                    # Build filter based on available information
                    if url:
                        # Most precise deletion using file_upload_id
                        filter_criteria = {"source": {"$eq": url}}
                        print(f"Using URL for precise deletion: {url}")
                    
                    # Delete vectors in the user's namespace that match the filter criteria
                    delete_result = index.delete(filter=filter_criteria, namespace=user_id)
                    print(f"Successfully deleted vectors from Pinecone for link '{url}' in namespace '{user_id}'.")
                    
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
                    raise HTTPException(status_code=500, detail="Failed to delete vectors from Pinecone.")
            except Exception as e:
                print(f"WARNING: Index '{index_name}' does not exist or is not accessible: {e}")
                print("Skipping vector deletion.")
        else:
            print("WARNING: Pinecone client not available. Skipping vector deletion.")

        return JSONResponse(
            status_code=200,
            content={
                "message": f"{url} was deleted successfully",
                "url": url,
                "user_id": user_id,
            }
        )
    except Exception as e:
        print(f"CRITICAL ERROR in delete_file_route: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file deletion.")