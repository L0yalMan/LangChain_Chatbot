from typing import Optional, List
from pydantic import BaseModel

class FileUpResponse(BaseModel):   
    file_name: str
    presigned_url: Optional[List[str]] = None
    s3_path: Optional[str] = None
    upload_id: Optional[str] = None
    part_num: Optional[List[int]] = None
    message: str
