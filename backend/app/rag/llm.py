# New llm pour le pipeline RAG (Retrieval-Augmented Generation). Il contient la fonction pour générer une réponse à partir d'un prompt système et d'un prompt utilisateur, en utilisant le modèle de langage Ollama. Elle envoie une requête HTTP à l'API Ollama et retourne le texte généré
import os

import httpx

from app.core.config import settings

NUM_CTX = int(os.getenv("LLM_NUM_CTX", "3072"))
MAX_ANSWER_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))

# Fonction pour générer une réponse à partir d'un prompt système et d'un prompt utilisateur. Elle envoie une requête HTTP à l'API Ollama et retourne le texte généré. Elle gère les erreurs de réponse et lève une exception si l'API ne répond pas correctement
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
            "keep_alive": "30m",
            "options": {
                "temperature": 0.1,
                "num_ctx": NUM_CTX,
                "num_predict": MAX_ANSWER_TOKENS,
            },
        },
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama a répondu {response.status_code} : {response.text[:500]}"
        )
    return response.json()["message"]["content"].strip()
