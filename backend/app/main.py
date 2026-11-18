# Ce fichier contient le code principal de l'application FastAPI. Il définit l'instance de l'application, les routes, les points de terminaison et la gestion du cycle de vie de l'application. Il importe également les modules nécessaires pour le routage, la gestion des fichiers statiques et la configuration des chemins d'accès aux répertoires.    
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
# On définit les chemins d'accès aux répertoires de l'application, tels que le répertoire backend, le répertoire frontend/web et le répertoire data/extracted. Le répertoire data/extracted est créé s'il n'existe pas déjà, afin de stocker les fichiers extraits par l'application
BACKEND_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BACKEND_DIR.parent / "frontend" / "web"
EXTRACTED_DIR = BACKEND_DIR / "data" / "extracted"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# On définit un gestionnaire de cycle de vie pour l'application FastAPI. Il est exécuté au démarrage et à l'arrêt de l'application. Au démarrage, il charge le modèle d'embedding une fois en appelant la fonction get_index() du module app.rag.retriever. Cela permet de préparer le modèle pour la récupération des passages pertinents lors des requêtes ultérieures.
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.rag.retriever import get_index

    get_index()  # charge le modèle d'embedding une fois, au démarrage
    yield

# On crée l'instance de l'application FastAPI avec le titre, la version et le gestionnaire de cycle de vie définis précédemment. On inclut le routeur API pour les points de terminaison de l'API version 1, et on monte les fichiers statiques pour les images extraites et l'interface utilisateur web si elle existe. On définit également un point de terminaison /health pour vérifier l'état de santé de l'application, et un point de terminaison racine / qui redirige vers l'interface utilisateur web ou la documentation Swagger selon la disponibilité.
app = FastAPI(title="MaintiDoc API", version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")
app.mount("/files", StaticFiles(directory=EXTRACTED_DIR), name="files")  # images extraites
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/" if WEB_DIR.exists() else "/docs")
