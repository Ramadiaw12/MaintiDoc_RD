# Transformation
# Charger le modèle d'intégration HuggingFace à partir des paramètres de configuration et le mettre en cache pour une utilisation ultérieure.
from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings

# Alors nous avons la fonction get_embed_model qui est décorée avec @lru_cache(maxsize=1) pour mettre en cache le modèle d'intégration HuggingFace.
# Elle retourne une instance de HuggingFaceEmbedding en utilisant le nom du modèle spécifié dans les paramètres de configuration. Cette fonction est utilisée pour obtenir le modèle d'intégration de manière efficace, en évitant de le recharger à chaque appel, 
# ce qui améliore les performances lors de l
@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=settings.embedding_model)