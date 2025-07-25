import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
# from supabase import create_client, Client

load_dotenv('.env') 

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
