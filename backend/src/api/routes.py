from fastapi import APIRouter, UploadFile, File, Request, Form, Depends, status

from src.utils.dependencies import get_current_user, TokenData # Adjust import based on your file structure

from src.core.models import ChatRequest
from src.controller.chat.chat import chat_with_rag
from src.controller.files.file_upload import upload_file
from src.controller.files.file_delete import delete_file
from src.controller.website.ingest_website import ingest_website_link
from src.controller.retrieval.configure_retrieval import configure_retrieval
from src.controller.retrieval.retrieval_config import get_retrieval_config

router = APIRouter()

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

@router.post("/chat/")
async def chat_with_rag_endpoint(
    request_data: ChatRequest,
    current_user: TokenData = Depends(get_current_user)
):
    return await chat_with_rag(request_data, current_user)

@router.post("/configure-retrieval/")
async def configure_retrieval_endpoint(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
):
    return await configure_retrieval(request)

@router.get("/retrieval-config/")
async def get_retrieval_config_endpoint():
    return await get_retrieval_config()