from supabase import create_client, Client
from src.core.config import settings

# Initialize the Supabase client
def create_supabase_client():
    try:
        supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        print("Supabase client created successfully!")
        return supabase
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        raise

# Usage example
supabase = create_supabase_client()