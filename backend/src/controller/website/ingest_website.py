from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma

from src.core.config import CHROMA_PATH, update_vector_store_globals
from src.core.retriever import create_advanced_retriever
from src.utils.document_processing import load_website_content, split_documents

async def ingest_website_link(request: Request):
    """
    Ingests content from a given website URL into the RAG system.
    It scrapes the content and adds it to the ChromaDB vector store.
    """
    from src.core.config import embeddings, vectorstore

    try:
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        url = body.get("url")
        user_id = body.get("userId")

        if not url or not url.strip():
            raise HTTPException(status_code=400, detail="Website URL not provided or empty.")

        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty.")

        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL format. URL must start with http:// or https://")

        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings not initialized. Please check your Google API key.")

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
        splits = split_documents(docs, total_content_length)

        if not splits:
            raise HTTPException(status_code=400, detail="No text chunks could be created from the website content.")

        print(f"--- CHUNKS CREATED FROM WEBSITE: {url} ---")
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
            print("--- APPENDING WEBSITE CONTENT TO EXISTING VECTOR STORE ---")
            vectorstore.add_documents(splits)
        else:
            print("--- CREATING NEW VECTOR STORE FROM WEBSITE CONTENT ---")
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
