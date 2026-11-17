from fastapi import APIRouter, HTTPException

from app.agent.agent import agent_answer
from app.schemas.agent import AgentRequest, AgentResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=AgentResponse)
def agent_chat(request: AgentRequest):
    try:
        history = [m.model_dump() for m in request.history]
        return agent_answer(request.question, history)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Erreur de l'agent : {error}") from error
