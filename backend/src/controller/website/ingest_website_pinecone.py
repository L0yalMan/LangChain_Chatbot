from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_pinecone import PineconeVectorStore

from src.core.config_pinecone import update_vector_store_globals
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

        vectorstore = vectorstores.get(user_id)
        if vectorstore:
            print("--- APPENDING WEBSITE CONTENT TO EXISTING VECTOR STORE ---")
            vectorstore.add_documents(split_docs)
        else:
            print("--- CREATING NEW VECTOR STORE FROM WEBSITE CONTENT ---")
            new_vectorstore = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text",
                namespace=user_id
            )
            update_vector_store_globals(user_id, new_vectorstore, None)
            vectorstore = new_vectorstore

        current_vectorstore = vectorstore
        new_retriever = create_advanced_retriever(current_vectorstore)
        if new_retriever is None:
            print("WARNING: Failed to create advanced retriever after website ingestion")
        else:
            update_vector_store_globals(current_vectorstore, new_retriever)
        
        try:
            collection_count = vectorstore._collection.count()
            print(f"--- VECTOR STORE UPDATED ---")
            print(f"Total chunks in vector store after website ingestion: {collection_count}")
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
