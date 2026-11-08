from fastapi import APIRouter, HTTPException

from app.rag.pipeline import answer
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return answer(request.question, top_k=request.top_k)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Erreur du pipeline RAG : {error}") from error
