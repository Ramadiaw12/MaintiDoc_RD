from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.rag.retriever import get_index

    get_index()  # charge le modèle d'embedding une fois, au démarrage
    yield


app = FastAPI(title="MaintiDoc API", version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
