from fastapi import APIRouter

from app.api.v1 import agent, chat, documents

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(agent.router)
api_router.include_router(documents.router)
