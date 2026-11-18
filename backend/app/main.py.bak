# Fichier principal de l'application FastAPI. Il définit le contexte de vie de l'application, les routes principales et les points de terminaison pour la santé et la documentation. Il importe le routeur de l'API v1 et inclut les routes dans l'application FastAPI. 
# Il utilise un gestionnaire de contexte asynchrone pour charger le modèle d'embedding au démarrage de l'application
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.v1.router import api_router

# Alors là on définit un gestionnaire de contexte asynchrone pour la durée de vie de l'application. Il importe la fonction `get_index` du module `retriever` et l'appelle une fois au démarrage de l'application pour charger le modèle d'embedding. 
# Cela permet d'optimiser les performances en évitant de recharger le modèle à chaque requête
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.rag.retriever import get_index

    get_index()  # charge le modèle d'embedding une fois, au démarrage
    yield

# On crée l'application FastAPI avec le titre "MaintiDoc API", la version "0.1.0" et le gestionnaire de contexte défini précédemment.
# On inclut le routeur de l'API v1 avec le préfixe "/api/v1". On définit également un point de terminaison "/health" pour vérifier l'état de l'application et un point de terminaison racine "/" qui redirige vers la documentation Swagger UI
app = FastAPI(title="MaintiDoc API", version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")

# On recupère le chemin de base de l'API à partir des paramètres de configuration et on définit un point de terminaison "/health" pour vérifier l'état de l'application. Il retourne un dictionnaire avec le statut "ok"
@app.get("/health")
def health():
    return {"status": "ok"}

# On définit un point de terminaison racine "/" qui redirige vers la documentation Swagger UI. Cela permet aux utilisateurs d'accéder facilement à la documentation interactive de l'API
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
