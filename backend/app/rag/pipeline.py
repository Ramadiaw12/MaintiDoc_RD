# File pour le pipeline RAG (Retrieval-Augmented Generation). 
# Il contient les fonctions pour construire le contexte à partir des passages récupérés, générer une réponse à partir de la question et des passages, et gérer les citations et les sources. Il utilise les modules `llm` pour la génération de texte, `prompts` pour les modèles de prompt, et `retriever` pour la récupération des passages pertinents
import re

from app.core.config import settings
from app.rag.llm import generate
from app.rag.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from app.rag.retriever import retrieve

# Variables pour la gestion des réponses et des types de passages. NOT_FOUND est le message retourné lorsque la réponse n'est pas trouvée dans les passages récupérés. TYPE_LABELS est un dictionnaire qui associe les types de passages à des étiquettes lisibles pour l'utilisateur
NOT_FOUND = "Je ne trouve pas cette information dans la documentation indexée."
TYPE_LABELS = {"text": "texte", "table": "tableau", "image": "description d'une figure"}

# Alors là on définit une fonction pour construire le contexte à partir des passages récupérés. Elle prend une liste de passages et retourne une chaîne de caractères formatée avec les numéros, les pages, les types et les extraits des passages. Cela permet de fournir un contexte clair et structuré pour la génération de la réponse
def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        label = TYPE_LABELS.get(c["type"], c["type"])
        parts.append(f"[{i}] (page {c['page']}, {label})\n{c['text']}")
    return "\n\n".join(parts)

# Ensuite, on définit la fonction principale pour obtenir la réponse du pipeline RAG. Elle prend une question et un nombre de passages à consulter. 
# Elle récupère les passages pertinents, construit le contexte, génère la réponse avec le modèle de langage, et gère les citations et les sources. Elle retourne un dictionnaire avec la réponse finale, si elle est sourcée, les citations valides et invalides, et les sources consultées
def answer(question: str, top_k: int = 5) -> dict:
    chunks = [c for c in retrieve(question, top_k) if c["score"] >= settings.min_score]

    if not chunks:
        return {"answer": NOT_FOUND, "grounded": False, "cited": [],
                "invalid_citations": [], "sources": []}

    prompt = USER_TEMPLATE.format(
        context=build_context(chunks), question=question, n=len(chunks)
    )
    text = generate(SYSTEM_PROMPT, prompt)

# Ici, on gère les citations et les sources dans la réponse générée. On extrait les numéros cités dans le texte, on vérifie leur validité par rapport aux passages disponibles, on retire les citations invalides, et on construit la liste des sources avec leurs informations. Enfin, on retourne un dictionnaire avec la réponse finale, si elle est sourcée, les citations valides et invalides, et les sources consultées
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    cited = sorted(n for n in numbers if 1 <= n <= len(chunks))
    invalid = sorted(n for n in numbers if n not in cited)
    for n in invalid:
        text = text.replace(f"[{n}]", "")  # on retire les sources inexistantes
    text = re.sub(r"[ \t]+", " ", text).strip()

# On construit la liste des sources avec leurs informations, en associant chaque numéro cité à son passage correspondant. On retourne un dictionnaire avec la réponse finale, si elle est sourcée, les citations valides et invalides, et les sources consultées
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
    return {"answer": text, "grounded": bool(cited), "cited": cited,
            "invalid_citations": invalid, "sources": sources}
