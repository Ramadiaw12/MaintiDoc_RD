from llama_index.vector_stores.postgres import PGVectorStore
# Fonction pour obtenir une instance de PGVectorStore avec les paramètres de connexion à la base de données PostgreSQL
def get_text_vector_store(embed_dim: int = 1024) -> PGVectorStore:
    return PGVectorStore.from_params(
        host="localhost", port=5432,
        database="maintidoc", user="maintidoc", password="maintidoc", # Paramètres de connexion à la base de données
        table_name="text_chunks",
        embed_dim=embed_dim,  # doit correspondre au modèle d'embedding choisi
    )