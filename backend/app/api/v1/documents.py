import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import text

from app.db.session import engine
from app.ingestion.pipeline import delete_document, ingest_pdf
from app.schemas.documents import DocumentOut, IngestResponse

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"
MAX_SIZE = 50 * 1024 * 1024  # 50 Mo


def safe_name(filename: str | None) -> str:
    name = Path(filename or "document.pdf").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


@router.post("", response_model=IngestResponse)
def upload_document(file: UploadFile = File(...), with_images: bool = True):
    name = safe_name(file.filename)
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    folder = UPLOAD_DIR / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name

    size, too_big = 0, False
    with open(target, "wb") as out:
        while block := file.file.read(1024 * 1024):
            size += len(block)
            if size > MAX_SIZE:
                too_big = True
                break
            out.write(block)
    if too_big:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 50 Mo).")

    with open(target, "rb") as f:
        is_pdf = f.read(5) == b"%PDF-"
    if not is_pdf:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Ce fichier n'est pas un PDF valide.")

    try:
        result = ingest_pdf(target, with_images=with_images)
    except Exception as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Échec de l'ingestion : {error}") from error

    if result["status"] == "already_indexed":
        shutil.rmtree(folder, ignore_errors=True)
    return result


@router.get("", response_model=list[DocumentOut])
def list_documents():
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, filename, page_count, created_at FROM documents ORDER BY id DESC")
        ).mappings().all()
    return [dict(r) for r in rows]


@router.delete("/{document_id}", status_code=204)
def remove_document(document_id: int):
    with engine.connect() as conn:
        found = conn.execute(
            text("SELECT 1 FROM documents WHERE id = :d"), {"d": document_id}
        ).first()
    if not found:
        raise HTTPException(status_code=404, detail="Document introuvable.")
    delete_document(document_id)
