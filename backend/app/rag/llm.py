# Le file est responsable de la génération de réponses à partir d'un modèle de langage (LLM) en utilisant l'API Ollama
import httpx

from app.core.config import settings

#  La fonction generate prend deux chaînes de caractères en entrée : system et user.
#  Elle envoie une requête POST à l'API Ollama pour générer une réponse basée sur le modèle de langage spécifié dans les paramètres de configuration. 
#  La requête inclut les messages du système et de l'utilisateur, ainsi que des options pour le modèle, retourne la réponse générée par le modèle après avoir vérifié que la requête a réussi.
def generate(system: str, user: str) -> str:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
