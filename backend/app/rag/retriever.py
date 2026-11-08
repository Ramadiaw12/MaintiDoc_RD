# Partie qui gère la récupération d'informations à partir d'un index vectoriel pour le pipeline de question-réponse. 
# Elle utilise un index vectoriel pour rechercher les documents pertinents en fonction d'une question posée par l'utilisateur, et retourne les résultats sous forme de chunks contenant le texte, la page, le type, la source et le chemin du fichier associé.
from functools import lru_cache

from llama_index.core import VectorStoreIndex

from app.db.vector_store import get_text_vector_store
from app.ingestion.embeddings import get_embed_model

# Ici, on a la fonction get_index qui est décorée avec @lru_cache(maxsize=1) pour mettre en cache l'index vectoriel.    
@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        get_text_vector_store(), embed_model=get_embed_model()
    )

# La fonction retrieve prend une question en entrée et un paramètre top_k pour spécifier le nombre de résultats à récupérer.
# Elle utilise l'index vectoriel pour rechercher les documents pertinents en fonction de la question
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
