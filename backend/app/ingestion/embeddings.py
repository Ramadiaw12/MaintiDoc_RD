# Transformation
from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings

# 
@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=settings.embedding_model)