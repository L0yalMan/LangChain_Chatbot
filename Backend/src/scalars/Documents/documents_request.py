from pydantic import BaseModel
from typing import List

class FileUpRequest(BaseModel):
    user_id: str
    part_num: int
    file_name: str

class parts(BaseModel):
    PartNumber: int
    ETag: str

class CompleteMultipartUploadRequest(BaseModel):
    user_id: str
    file_name: str
    upload_id: str
    parts: List[parts]

class DeleteFileRequest(BaseModel):
    file_type: str
    file_id: str
    user_id: str

class UpdateFileTagRequest(BaseModel):
    file_tag: str
    file_ids: List[str]
    user_id: str