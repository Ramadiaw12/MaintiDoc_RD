from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import make_url

from app.core.config import settings


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