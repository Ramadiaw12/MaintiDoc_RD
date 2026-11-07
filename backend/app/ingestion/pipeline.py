# Enchaîne le tout , avec une transaction SQL pour enregistrer le document et ses images, puis découpe les documents textuels et tabulaires en nœuds de texte, calcule leurs embeddings et les stocke dans le magasin vectoriel PGVectorStore.

from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from sqlalchemy import text

from app.db.session import engine
from app.db.vector_store import get_text_vector_store
from app.ingestion.chunker import build_nodes
from app.ingestion.embeddings import get_embed_model
from app.ingestion.parser import compute_hash, parse_pdf

# Extraction 
EXTRACTED_DIR = Path(__file__).resolve().parents[2] / "data" / "extracted"

# Fonction ingest_pdf qui prend en entrée le chemin d'un fichier PDF, vérifie s'il a déjà été indexé en consultant la base de données, puis extrait les informations du PDF, enregistre le document et ses images dans la base de données, découpe les documents textuels et tabulaires en nœuds de texte, calcule leurs embeddings et les stocke dans le magasin vectoriel PGVectorStore. La fonction retourne un dictionnaire contenant le statut de l'opération, l'identifiant du document, le nombre de nœuds créés et le nombre d'images extraites.
def ingest_pdf(pdf_path: str | Path) -> dict:
    pdf_path = Path(pdf_path)
    file_hash = compute_hash(pdf_path)

    # 1. Ce PDF est-il déjà indexé ?
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT id FROM documents WHERE file_hash = :h"), {"h": file_hash}
        ).first()
    if existing:
        return {"status": "already_indexed", "document_id": existing.id}

    # 2. Extraction
    parsed = parse_pdf(pdf_path, EXTRACTED_DIR / file_hash[:12])

    # 3. Enregistrer le document et ses images
    with engine.begin() as conn:
        document_id = conn.execute(
            text("INSERT INTO documents (filename, file_hash, page_count) "
                 "VALUES (:f, :h, :p) RETURNING id"),
            {"f": parsed.filename, "h": file_hash, "p": parsed.page_count},
        ).scalar_one()

        for img in parsed.images:
            conn.execute(
                text("INSERT INTO images (document_id, page, file_path) "
                     "VALUES (:d, :p, :f)"),
                {"d": document_id, "p": img["page"], "f": img["file_path"]},
            )

    # 4. Découpage, embeddings et stockage dans pgvector
    nodes = build_nodes(parsed, document_id)
    storage_context = StorageContext.from_defaults(vector_store=get_text_vector_store())
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(),
        show_progress=True,
    )

    return {
        "status": "indexed",
        "document_id": document_id,
        "chunks": len(nodes),
        "images": len(parsed.images),
    }