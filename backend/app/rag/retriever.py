from functools import lru_cache

from llama_index.core import VectorStoreIndex

from app.db.vector_store import get_text_vector_store
from app.ingestion.embeddings import get_embed_model


@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        get_text_vector_store(), embed_model=get_embed_model()
    )


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    results = get_index().as_retriever(similarity_top_k=top_k).retrieve(question)
    chunks = []
    for r in results:
        meta = r.node.metadata
        chunks.append({
            "score": r.score or 0.0,
            "text": r.node.get_content(),
            "page": meta.get("page"),
            "type": meta.get("type"),
            "source": meta.get("source"),
            "file_path": meta.get("file_path"),
        })
    return chunks
