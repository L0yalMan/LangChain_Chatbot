import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request, HTTPException

# Import configuration and routes
from src.core.config_pinecone import initialize_llm_embeddings_pinecone, get_global_state, Settings
from src.api.routes import router
from src.core.retriver_pinecone import initialize_vector_store

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
    title="RAG System with Pinecone, LangGraph, and FastAPI",
    description="An API for uploading documents and chatting with a multi-tenant RAG pipeline.",
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
        # Initialize LLM, embeddings, and Pinecone client globally
        init_success = initialize_llm_embeddings_pinecone()
        
        if init_success:
            print("SUCCESS: LLM, embeddings, and Pinecone client initialized.")
        else:
            print("WARNING: LLM, embeddings, or Pinecone client initialization failed.")
            print("Please check your GOOGLE_API_KEY and PINECONE_API_KEY environment variables.")
        
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
        return {"message": "LangChain RAG API with Pinecone is running!"}
    except Exception as e:
        print(f"CRITICAL ERROR in read_root: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/debug/globals/{user_id}")
def debug_globals(user_id: str):
    """Debug endpoint to check global variable states for a specific user."""
    from src.core.config_pinecone import get_global_state
    try:
        global_state = get_global_state(user_id)
        return {
            "message": f"Global variable states for user: {user_id}",
            "global_state": global_state
        }
    except Exception as e:
        print(f"ERROR in debug_globals: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while checking global state.")
