from datetime import datetime

from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    document_id: int
    chunks: int | None = None
    images: int | None = None


class DocumentOut(BaseModel):
    id: int
    filename: str
    page_count: int | None = None
    created_at: datetime
