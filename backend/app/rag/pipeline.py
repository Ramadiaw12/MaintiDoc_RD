# Mettre en place un pipeline de question-réponse basé sur la récupération d'informations à partir d'un index vectoriel. Le pipeline prend une question en entrée, récupère les documents pertinents à partir de l'index, construit un contexte à partir de ces documents, et génère une réponse en utilisant un modèle de langage. 
# La réponse est ensuite analysée pour identifier les références aux documents cités et fournir des informations sur les sources utilisées.
import re

from app.core.config import settings
from app.rag.llm import generate
from app.rag.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from app.rag.retriever import retrieve

NOT_FOUND = "Je ne trouve pas cette information dans la documentation indexée."
TYPE_LABELS = {"text": "texte", "table": "tableau", "image": "description d'une figure"}

# Alors là , la fonction build_context prend une liste de chunks (dictionnaires contenant des informations sur les documents récupérés) et construit un contexte textuel à partir de ces chunks. 
def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        label = TYPE_LABELS.get(c["type"], c["type"])
        parts.append(f"[{i}] (page {c['page']}, {label})\n{c['text']}")
    return "\n\n".join(parts)

# FOnction principale du pipeline de question-réponse. Elle prend une question en entrée, récupère les chunks pertinents à partir de l'index vectoriel, construit un contexte à partir de ces chunks, et génère une réponse en utilisant un modèle de langage.
def answer(question: str, top_k: int = 5) -> dict:
    chunks = [c for c in retrieve(question, top_k) if c["score"] >= settings.min_score]

    if not chunks:
        return {"answer": NOT_FOUND, "grounded": False, "cited": [], "sources": []}

    prompt = USER_TEMPLATE.format(context=build_context(chunks), question=question)
    text = generate(SYSTEM_PROMPT, prompt)

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
    sources = [
        {
            "n": i,
            "page": c["page"],
            "type": c["type"],
            "score": round(c["score"], 3),
            "file_path": c["file_path"],
            "extract": c["text"][:200],
        }
        for i, c in enumerate(chunks, start=1)
    ]
    return {"answer": text, "grounded": bool(cited), "cited": cited, "sources": sources}
