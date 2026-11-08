# Lie la base de données PostgreSQL et le magasin vectoriel PGVectorStore pour stocker les nœuds de texte et leurs intégrations.
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import make_url

from app.core.config import settings

# Alors là, la fonction get_text_vector_store retourne une instance de PGVectorStore configurée pour se connecter à la base de données PostgreSQL en utilisant les paramètres de configuration. 
# Elle utilise l'URL de la base de données pour extraire les informations d'hôte, de port, de nom de base de données, d'utilisateur et de mot de passe, puis crée un magasin vectoriel PGVectorStore avec ces paramètres et le nom de la table "text_chunks" ainsi que la dimension des intégrations spécifiée dans les paramètres de configuration. 
# Cette fonction est utilisée pour obtenir un magasin vectoriel pour stocker les nœuds de texte et leurs intégrations dans la base de données PostgreSQL.
def get_text_vector_store() -> PGVectorStore:
    url = make_url(settings.database_url)
    return PGVectorStore.from_params(
        host=url.host,
        port=url.port,
        database=url.database,
        user=url.username,
        password=url.password,
        table_name="text_chunks",
        embed_dim=settings.embedding_dim, 
    )