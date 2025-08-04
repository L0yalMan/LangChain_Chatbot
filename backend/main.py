import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request, HTTPException

# Import configuration and routes
from src.core.config import CHROMA_PATH, initialize_llm_embeddings, get_global_state
from src.api.routes import router
from src.core.retriever import initialize_vector_store

# --- Global Variables & Setup ---
load_dotenv()

# Set default environment variables to prevent warnings
os.environ.setdefault("USER_AGENT", "LangChain-RAG-Bot/1.0.0")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("CURL_CA_BUNDLE", "")

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*USER_AGENT.*")
warnings.filterwarnings("ignore", message=".*unclosed.*")


# FastAPI App
app = FastAPI(
    title="RAG System with ChromaDB, LangGraph, and FastAPI",
    description="An API for uploading documents and chatting with a RAG pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch any unhandled exceptions."""
    try:
        print(f"UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}")
        print(f"Request URL: {request.url}")
        print(f"Request method: {request.method}")

        # Log additional context if available
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    print(f"Request body: {body[:500]}...")  # First 500 chars
        except Exception as e:
            print(f"Could not read request body: {e}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "type": type(exc).__name__
            }
        )
    except Exception as handler_error:
        # If the exception handler itself fails, return a simple error response
        print(f"CRITICAL: Exception handler failed: {handler_error}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred."
            }
        )

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("--- APPLICATION STARTUP ---")
    try:
        # Ensure the chroma_db directory exists
        os.makedirs(CHROMA_PATH, exist_ok=True)
        print(f"Ensured chroma_db directory exists: {CHROMA_PATH}")

        # Initialize LLM and Embeddings globally
        init_success = initialize_llm_embeddings()
        
        if init_success:
            # Initialize vector store after embeddings are ready
            initialize_vector_store()
            
            # Check global state after initialization
            global_state = get_global_state()
            print(f"DEBUG: Global state after initialization: {global_state}")
            
            # Import the updated global variables after initialization
            from src.core.config import vectorstore, retriever
            
            # Verify that global variables were properly updated
            if vectorstore is not None:
                try:
                    collection_count = vectorstore._collection.count()
                    print(f"SUCCESS: Vector store initialized with {collection_count} documents")
                except Exception as e:
                    print(f"SUCCESS: Vector store initialized but could not get collection count: {e}")
            else:
                print("WARNING: Vector store initialization failed - global variable is None")
                
            if retriever is not None:
                print("SUCCESS: Retriever initialized successfully")
            else:
                print("WARNING: Retriever initialization failed - global variable is None")
        else:
            print("WARNING: LLM and embeddings initialization failed. Vector store will not be available.")
            print("Please set the GOOGLE_API_KEY environment variable to enable full functionality.")

        print("--- APPLICATION STARTUP COMPLETE ---")
    except Exception as e:
        print(f"Error during startup: {e}")
        # Don't raise the exception, just log it

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("--- APPLICATION SHUTDOWN ---")
    try:
        # Clean up any resources if needed
        print("Cleaning up resources...")
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        print("--- APPLICATION SHUTDOWN COMPLETE ---")

# Include API routes
app.include_router(router)

@app.get("/")
def read_root():
    try:
        return {"message": "LangChain RAG API is running!"}
    except Exception as e:
        print(f"CRITICAL ERROR in read_root: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/debug/globals")
def debug_globals():
    """Debug endpoint to check global variable states."""
    try:
        global_state = get_global_state()
        return {
            "message": "Global variable states",
            "global_state": global_state
        }
    except Exception as e:
        print(f"ERROR in debug_globals: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while checking global state.")