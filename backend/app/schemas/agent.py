from pydantic import BaseModel, Field

from app.schemas.chat import ChatResponse


class HistoryMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)


class AgentResponse(ChatResponse):
    mode: str
    tools_used: list[str] = []
    fallback_reason: str | None = None
