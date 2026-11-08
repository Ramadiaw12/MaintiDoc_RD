# Le chunker doit transformer chaque description en chunk de type image, qui garde le chemin de l’image en métadonnée
# TRaitement d'images : on va générer un vecteur CLIP pour chaque image, et l'enregistrer dans la base de données. On va aussi générer un texte descriptif pour chaque image, et l'enregistrer dans la base de données
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

# Here we have the function to_pgvector that takes a list of floats (representing a vector) or None as input. If the input is None, it returns None. Otherwise, it formats the list of floats into a string representation suitable for PostgreSQL's vector type, with each float rounded to six decimal places and enclosed in square brackets
def to_pgvector(vector: list[float] | None) -> str | None:
    if vector is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"

# ALors là , la fonction delete_document prend un identifiant de document en entrée et supprime le document correspondant de la base de données, ainsi que ses images associées (en cascade) et ses chunks. Elle utilise une transaction pour s'assurer que toutes les suppressions sont effectuées de manière atomique
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

# Alors là, la fonction ingest_pdf prend le chemin d'un fichier PDF et un booléen indiquant si les images doivent être traitées. Elle vérifie d'abord si le PDF a déjà été indexé en calculant son hachage et en recherchant dans la base de données. Si le PDF n'est pas encore indexé, il est analysé pour extraire le texte, les tableaux et les images. Les images sont ensuite traitées pour générer des vecteurs CLIP et des descriptions textuelles. Le document et les images sont enregistrés dans la base de données, puis les chunks sont créés à partir du contenu extrait et indexés avec leurs embeddings. 
# En cas d'erreur lors de l'indexation des chunks, le document est supprimé pour éviter des incohérences dans la base de données.
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
