from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    n: int
    page: int | None = None
    type: str | None = None
    score: float
    file_path: str | None = None
    extract: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    cited: list[int]
    invalid_citations: list[int]
    sources: list[Source]
