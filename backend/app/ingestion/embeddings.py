# Transformation
# Charger le modèle d'intégration HuggingFace à partir des paramètres de configuration et le mettre en cache pour une utilisation ultérieure.
from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings

# 
@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=settings.embedding_model)