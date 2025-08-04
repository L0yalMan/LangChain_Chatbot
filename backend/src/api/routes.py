from fastapi import APIRouter, UploadFile, File, Request, Form

from src.core.models import ChatRequest
from src.controller.chat.chat import chat_with_rag
from src.controller.files.file_upload import upload_file
from src.controller.files.file_delete import delete_file
from src.controller.website.ingest_website import ingest_website_link
from src.controller.retrieval.configure_retrieval import configure_retrieval
from src.controller.retrieval.retrieval_config import get_retrieval_config

router = APIRouter()

@router.post("/files/delete")
async def delete_file_endpoint(request: Request):
    return await delete_file(request)

@router.post("/files/upload/")
async def upload_file_endpoint(file: UploadFile = File(...), user_id: str = Form(default="anonymous")):
    try:
        return await upload_file(file, user_id)
    except Exception as e:
        print(f"Error in upload_file_endpoint: {e}")
        raise

@router.post("/ingest-website/")
async def ingest_website_link_endpoint(request: Request):
    return await ingest_website_link(request)

@router.post("/chat/")
async def chat_with_rag_endpoint(request_data: ChatRequest):
    return await chat_with_rag(request_data)

@router.post("/configure-retrieval/")
async def configure_retrieval_endpoint(request: Request):
    return await configure_retrieval(request)

@router.get("/retrieval-config/")
async def get_retrieval_config_endpoint():
    return await get_retrieval_config()