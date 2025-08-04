from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from src.core.config import RETRIEVAL_CONFIG, update_vector_store_globals
from src.core.retriever import create_advanced_retriever

async def configure_retrieval(request: Request):
    """Configure retrieval parameters dynamically."""
    from src.core.config import vectorstore

    try:
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        if not body:
            raise HTTPException(status_code=400, detail="Request body is empty.")

        try:
            if 'default_k' in body:
                value = body['default_k']
                if not isinstance(value, int) or value <= 0 or value > 100:
                    raise HTTPException(status_code=400, detail="default_k must be a positive integer between 1 and 100.")
                RETRIEVAL_CONFIG['default_k'] = value

            if 'mmr_k' in body:
                value = body['mmr_k']
                if not isinstance(value, int) or value <= 0 or value > 100:
                    raise HTTPException(status_code=400, detail="mmr_k must be a positive integer between 1 and 100.")
                RETRIEVAL_CONFIG['mmr_k'] = value

            if 'mmr_lambda' in body:
                value = body['mmr_lambda']
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise HTTPException(status_code=400, detail="mmr_lambda must be a number between 0 and 1.")
                RETRIEVAL_CONFIG['mmr_lambda'] = value

            if 'similarity_threshold' in body:
                value = body['similarity_threshold']
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise HTTPException(status_code=400, detail="similarity_threshold must be a number between 0 and 1.")
                RETRIEVAL_CONFIG['similarity_threshold'] = value

            if 'max_chunk_size' in body:
                value = body['max_chunk_size']
                if not isinstance(value, int) or value <= 0 or value > 10000:
                    raise HTTPException(status_code=400, detail="max_chunk_size must be a positive integer between 1 and 10000.")
                RETRIEVAL_CONFIG['max_chunk_size'] = value

            if 'chunk_overlap' in body:
                value = body['chunk_overlap']
                if not isinstance(value, int) or value < 0 or value > 5000:
                    raise HTTPException(status_code=400, detail="chunk_overlap must be a non-negative integer between 0 and 5000.")
                RETRIEVAL_CONFIG['chunk_overlap'] = value

        except HTTPException:
            raise
        except Exception as e:
            print(f"ERROR: Failed to validate configuration parameters: {e}")
            raise HTTPException(status_code=400, detail="Invalid configuration parameters.")

        if vectorstore:
            new_retriever = create_advanced_retriever(vectorstore)
            if new_retriever is None:
                print("WARNING: Failed to create retriever with new configuration")
            else:
                update_vector_store_globals(vectorstore, new_retriever)
                print(f"Retriever reconfigured with new parameters: {RETRIEVAL_CONFIG}")
        else:
            print("WARNING: No vector store available, configuration saved but not applied")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Retrieval configuration updated successfully",
                "current_config": RETRIEVAL_CONFIG
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in configure_retrieval_route: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating configuration.")
