# app/dependencies.py
from fastapi import HTTPException, Header, status, Depends
from jose import jwt, JWTError
from pydantic import BaseModel
from typing import Optional

# Import the secret from your main config/app file
from src.core.config_pinecone import settings # Adjust this import based on your project structure

class TokenData(BaseModel):
    user_id: str
    email: Optional[str] = None
    # Add other claims you expect from your Supabase JWT
    # e.g., role: Optional[str] = None

async def get_current_user(authorization: Optional[str] = Header(None)) -> TokenData:
    """
    Dependency to verify Supabase JWT and extract user data.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, token = authorization.split(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization scheme must be Bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Decode and verify the JWT
        # The algorithm is usually HS256 for Supabase JWTs
        payload = jwt.decode(token, settings.SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")

        user_id: str = payload.get("sub") # 'sub' claim typically holds the user ID
        email: Optional[str] = payload.get("email") # 'email' claim

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials: User ID not found in token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Return a TokenData object containing validated user info
        print(f"----------->>>>>>>>user_id", user_id)
        print(f"----------->>>>>>>>email", email)
        return TokenData(user_id=user_id, email=email)

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials: Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Catch any other unexpected errors during token processing
        print(f"Error during JWT processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication.",
        )