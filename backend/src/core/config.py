import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from supabase import create_client, Client

load_dotenv('.env')  # This is already loaded in main.py 

class Settings(BaseSettings):
    # PROJECT_TITLE: str = ""
    # PROJECT_VERSION: str = "0.0.1"
    # HOST_HTTP: str = os.environ.get("HOST_HTTP", "http://")
    # HOST_URL: str = os.environ.get("HOST_URL", "localhost")
    # HOST_PORT: int = int(os.environ.get("HOST_PORT", "5000"))
    # BASE_URL: str = HOST_HTTP+HOST_URL+":"+str(HOST_PORT)
    
    # Supabase settings
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
    SUPABASE_JWT_SECRET: str = os.environ.get("SUPABASE_JWT_SECRET", "")
    SUPABASE_PROJECT_ID: str = os.environ.get("SUPABASE_PROJECT_ID", "")

    # AWS settings
    AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    AWS_S3_BUCKET_NAME: str = os.environ.get("AWS_S3_BUCKET_NAME", "")
    AWS_REGION: str = os.environ.get("AWS_REGION", "")

    # OpenAI and LLM settings
    # OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    # OPENAI_API_URL: str = os.environ.get("OPENAI_API_URL", "")
    # OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
    # OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "")

    # Gemini_API_KEY
    # GOOGLE_API_KEY = os.

    # Search API settings
    # SERPER_API_KEY: str = os.environ.get("SERPER_API_KEY", "")
    # SERPER_API_URL: str = os.environ.get("SERPER_API_URL", "")
    
    # Pinecone settings
    # PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    # USER_PINECONE_INDEX_NAME: str = os.environ.get("USER_PINECONE_INDEX_NAME", "")

    # Agents settings
    # APP_ENV: str = os.environ.get("APP_ENV", "development")
    # LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    # MAX_UPLOAD_SIZE: int = int(os.environ.get("MAX_UPLOAD_SIZE", "10"))
    # LOCAL_STORAGE_DIR: str = os.environ.get("LOCAL_STORAGE_DIR", "./local_storage")
    
    # LangSmith settings
    # LANGSMITH_TRACING: bool = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
    # LANGSMITH_ENDPOINT: str = os.environ.get("LANGSMITH_ENDPOINT", "")
    # LANGSMITH_API_KEY: str = os.environ.get("LANGSMITH_API_KEY", "")
    # LANGSMITH_PROJECT: str = os.environ.get("LANGSMITH_PROJECT", "")

    # LLAMA_KEY: str = os.environ.get("LLAMA_KEY", "")

    # @property
    # def supabase(self) -> Client:
    #     return create_client(
    #         self.SUPABASE_URL,
    #         self.SUPABASE_KEY
    #     )

settings = Settings()

CHROMA_PATH = "chroma_db"

llm = None
embeddings = None

vectorstore: Chroma = None
retriever = None

def initialize_llm_embeddings():
    global llm, embeddings
    try:
        # Check if GOOGLE_API_KEY is set
        google_api_key = os.getenv("GOOGLE_API_KEY")
        print(f"DEBUG: GOOGLE_API_KEY found: {'Yes' if google_api_key else 'No'}")
        if google_api_key:
            print(f"DEBUG: GOOGLE_API_KEY length: {len(google_api_key)}")
            print(f"DEBUG: GOOGLE_API_KEY starts with: {google_api_key[:10]}...")
        
        if google_api_key is None:
            print("ERROR: GOOGLE_API_KEY environment variable is not set. Cannot initialize LLM or embeddings.")
            llm = None
            embeddings = None
            return False
        
        print("DEBUG: Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
        )
        print("DEBUG: Initializing embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        )
        print("LLM and embeddings initialized successfully")
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM or embeddings: {e}")
        print(f"ERROR: Exception type: {type(e).__name__}")
        llm = None
        embeddings = None
        return False

def get_embeddings():
    """Get the embeddings instance, ensuring it's initialized."""
    global embeddings
    if embeddings is None:
        print("WARNING: Embeddings not initialized, attempting to initialize now...")
        initialize_llm_embeddings()
    return embeddings

def update_vector_store_globals(new_vectorstore, new_retriever):
    """Update the global vectorstore and retriever variables."""
    global vectorstore, retriever
    vectorstore = new_vectorstore
    retriever = new_retriever
    print(f"DEBUG: Updated global variables - vectorstore: {vectorstore is not None}, retriever: {retriever is not None}")

def get_global_state():
    """Get the current state of global variables for debugging."""
    global vectorstore, retriever, llm, embeddings
    return {
        'vectorstore': vectorstore is not None,
        'retriever': retriever is not None,
        'llm': llm is not None,
        'embeddings': embeddings is not None
    }

RETRIEVAL_CONFIG: Dict[str, Any] = {
    'default_k': 8,
    'mmr_k': 12,
    'mmr_lambda': 0.7,
    'similarity_threshold': 0.5,
    'max_chunk_size': 1500,
    'chunk_overlap': 300,
}
