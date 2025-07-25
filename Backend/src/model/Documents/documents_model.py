from typing import List, Optional, Dict, Any
from datetime import datetime
from dateutil import parser

class File:
    def __init__(
        self,
        id: str,
        user_id: str,
        filename: str,
        original_filename: str,
        file_type: str,
        file_size: int,
        path: str,
        created_at: datetime,
    ):
        self.id = id
        self.user_id = user_id
        self.filename = filename
        self.original_filename = original_filename
        self.file_type = file_type
        self.file_size = file_size
        self.path = path
        self.created_at = created_at if isinstance(created_at, datetime) else parser.parse(str(created_at))

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]):
        created_at = row['created_at']
        if isinstance(created_at, str):
            created_at = parser.parse(created_at)
            
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            filename=row['filename'],
            original_filename=row['original_filename'],
            file_type=row['file_type'],
            file_size=row['file_size'],
            path=row['path'],
            created_at=created_at,
        )

    def as_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "path": self.path,
            "created_at": self.created_at,
        }

