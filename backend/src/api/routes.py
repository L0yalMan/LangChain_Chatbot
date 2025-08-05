from fastapi import APIRouter, UploadFile, File, Request, Form, Depends, status

from src.utils.dependencies import get_current_user, TokenData # Adjust import based on your file structure

from src.core.models import ChatRequest
from src.controller.chat.chat_pinecone import chat_with_rag
from src.controller.files.file_upload_pinecone import upload_file
from src.controller.files.file_delete_pinecone import delete_file
from src.controller.website.ingest_website_pinecone import ingest_website_link, delete_website_link
from src.core.retriver_pinecone import initialize_vector_store
from src.core.config_pinecone import get_global_state

router = APIRouter()

@router.get("/init_user")
async def init_user_endpoint(
    current_user: TokenData = Depends(get_current_user)
):
    if get_global_state(current_user.user_id)['vectorstore'] is False:
        print("Initializing vector store")
        initialize_vector_store(current_user.user_id)
    else:
        print("User already initialized")
    return {"message": "User initialized successfully"}

@router.post("/files/delete")
async def delete_file_endpoint(
  request: Request, 
  current_user: TokenData = Depends(get_current_user) # This applies the authentication)
):
    return await delete_file(request, current_user)

@router.post("/files/upload/")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user)                         
):
    return await upload_file(current_user, file)

@router.post("/ingest-website/")
async def ingest_website_link_endpoint(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
):
    return await ingest_website_link(request, current_user)

@router.post("/delete-website/")
async def delete_website_link_endpoint(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
):
    return await delete_website_link(request, current_user)

@router.post("/chat/")
async def chat_with_rag_endpoint(
    request_data: ChatRequest,
    current_user: TokenData = Depends(get_current_user)
):
    return await chat_with_rag(request_data, current_user)