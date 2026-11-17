from langchain.tools import tool
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine
from app.ingestion.image_processor import embed_text_clip
from app.rag.retriever import retrieve

TYPE_LABELS = {"text": "texte", "table": "tableau", "image": "description d'une figure"}
MAX_CHARS = 800  # un passage trop long saturerait le contexte du petit modèle


def build_tools(registry: list[dict]) -> list:
    """Crée les outils d'UNE requête. `registry` collecte les sources numérotées."""

    def register(source: dict) -> int:
        n = len(registry) + 1
        registry.append({"n": n, **source})
        return n

    @tool
    def search_documentation(query: str) -> str:
        """Cherche dans la documentation indexée (texte, tableaux, descriptions de figures).
        À utiliser pour toute question technique. L'entrée est une courte requête,
        écrite dans la langue du document."""
        chunks = [c for c in retrieve(query, 3) if c["score"] >= settings.min_score]
        if not chunks:
            return "Aucun passage pertinent trouvé."
        parts = []
        for c in chunks:
            n = register({
                "page": c["page"], "type": c["type"], "score": round(c["score"], 3),
                "file_path": c["file_path"], "extract": c["text"][:200],
            })
            label = TYPE_LABELS.get(c["type"], c["type"])
            parts.append(f"[{n}] (page {c['page']}, {label})\n{c['text'][:MAX_CHARS]}")
        return "\n\n".join(parts)

    @tool
    def search_figures(description: str) -> str:
        """Cherche des figures, schémas ou images à partir d'une description de leur contenu.
        À utiliser quand l'utilisateur demande de voir un schéma, un diagramme ou une photo."""
        vector = embed_text_clip(description)
        vector_text = "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT page, file_path, caption, "
                     "1 - (clip_embedding <=> CAST(:q AS vector)) AS score "
                     "FROM images WHERE clip_embedding IS NOT NULL "
                     "ORDER BY clip_embedding <=> CAST(:q AS vector) LIMIT 2"),
                {"q": vector_text},
            ).mappings().all()
        if not rows:
            return "Aucune figure indexée."
        parts = []
        for r in rows:
            caption = r["caption"] or "Pas de description disponible."
            n = register({
                "page": r["page"], "type": "image", "score": round(float(r["score"]), 3),
                "file_path": r["file_path"], "extract": caption[:200],
            })
            parts.append(f"[{n}] (page {r['page']}, figure)\n{caption[:MAX_CHARS]}")
        return "\n\n".join(parts)

    @tool
    def list_documents() -> str:
        """Liste les documents indexés avec leur nombre de pages."""
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT filename, page_count FROM documents ORDER BY id")
            ).all()
        if not rows:
            return "Aucun document indexé."
        return "\n".join(f"- {r.filename} ({r.page_count} pages)" for r in rows)

    return [search_documentation, search_figures, list_documents]
