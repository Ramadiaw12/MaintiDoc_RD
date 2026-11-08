import httpx

from app.core.config import settings


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
