from fastapi import HTTPException
from fastapi.responses import JSONResponse

from src.core.config import RETRIEVAL_CONFIG

async def get_retrieval_config():
    """Get current retrieval configuration."""
    from src.core.config import vectorstore
    try:
        vectorstore_status = "loaded" if vectorstore else "not_loaded"
        total_chunks = 0

        try:
            if vectorstore:
                total_chunks = vectorstore._collection.count()
        except Exception as e:
            print(f"ERROR: Failed to get collection count: {e}")
            vectorstore_status = "error"

        return JSONResponse(
            status_code=200,
            content={
                "current_config": RETRIEVAL_CONFIG,
                "vectorstore_status": vectorstore_status,
                "total_chunks": total_chunks
            }
        )
    except Exception as e:
        print(f"CRITICAL ERROR in get_retrieval_config_route: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving configuration.")