from sqlalchemy import text
import httpx

from app.core.config import settings
from app.db.session import engine

# La fonction check_setup.py vérifie la configuration de l'application en testant la connexion à la base de données PostgreSQL, la présence de l'extension pgvector et la disponibilité du service Ollama
with engine.connect() as conn:
    print("PostgreSQL :", conn.execute(text("SELECT 1")).scalar() == 1)
    print("pgvector   :", conn.execute(
        text("SELECT extname FROM pg_extension WHERE extname='vector'")
    ).scalar())

# Vérification de la disponibilité du service Ollama en effectuant une requête HTTP GET à l'URL de base d'Ollama définie dans les paramètres de configuration. 
# La réponse est ensuite analysée pour afficher les modèles disponibles
r = httpx.get(f"{settings.ollama_base_url}/api/tags")
print("Ollama     :", [m["name"] for m in r.json()["models"]])