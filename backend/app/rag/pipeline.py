import re

from app.core.config import settings
from app.rag.llm import generate
from app.rag.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from app.rag.retriever import retrieve

NOT_FOUND = "Je ne trouve pas cette information dans la documentation indexée."
TYPE_LABELS = {"text": "texte", "table": "tableau", "image": "description d'une figure"}


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        label = TYPE_LABELS.get(c["type"], c["type"])
        parts.append(f"[{i}] (page {c['page']}, {label})\n{c['text']}")
    return "\n\n".join(parts)


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
