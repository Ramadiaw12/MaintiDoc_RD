from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from sqlalchemy import text

from app.db.session import engine
from app.db.vector_store import get_text_vector_store
from app.ingestion.chunker import build_nodes
from app.ingestion.embeddings import get_embed_model
from app.ingestion.image_processor import process_images
from app.ingestion.parser import compute_hash, parse_pdf

EXTRACTED_DIR = Path(__file__).resolve().parents[2] / "data" / "extracted"


def to_pgvector(vector: list[float] | None) -> str | None:
    if vector is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def delete_document(document_id: int) -> None:
    """Supprime un document, ses images (cascade) ET ses chunks."""
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT to_regclass('public.data_text_chunks')")
        ).scalar()
        if table_exists:
            conn.execute(
                text("DELETE FROM data_text_chunks "
                     "WHERE CAST(metadata_->>'document_id' AS integer) = :d"),
                {"d": document_id},
            )
        conn.execute(text("DELETE FROM documents WHERE id = :d"), {"d": document_id})


def ingest_pdf(pdf_path: str | Path, with_images: bool = True) -> dict:
    pdf_path = Path(pdf_path)
    file_hash = compute_hash(pdf_path)

    # 1. Déjà indexé ?
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT id FROM documents WHERE file_hash = :h"), {"h": file_hash}
        ).first()
    if existing:
        return {"status": "already_indexed", "document_id": existing.id}

    # 2. Extraction, puis description + vecteur CLIP de chaque image
    parsed = parse_pdf(pdf_path, EXTRACTED_DIR / file_hash[:12])
    if with_images:
        parsed.images = process_images(parsed.images)

    # 3. Document + images en base
    with engine.begin() as conn:
        document_id = conn.execute(
            text("INSERT INTO documents (filename, file_hash, page_count) "
                 "VALUES (:f, :h, :p) RETURNING id"),
            {"f": parsed.filename, "h": file_hash, "p": parsed.page_count},
        ).scalar_one()

        for img in parsed.images:
            conn.execute(
                text("INSERT INTO images (document_id, page, file_path, caption, clip_embedding) "
                     "VALUES (:d, :p, :f, :c, CAST(:e AS vector))"),
                {"d": document_id, "p": img["page"], "f": img["file_path"],
                 "c": img.get("caption"), "e": to_pgvector(img.get("clip_embedding"))},
            )

    # 4. Chunks + embeddings + pgvector (annulation complète en cas d'erreur)
    try:
        nodes = build_nodes(parsed, document_id)
        storage_context = StorageContext.from_defaults(vector_store=get_text_vector_store())
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=get_embed_model(),
            show_progress=True,
        )
    except Exception:
        delete_document(document_id)
        raise

    return {
        "status": "indexed",
        "document_id": document_id,
        "chunks": len(nodes),
        "images": len(parsed.images),
    }
