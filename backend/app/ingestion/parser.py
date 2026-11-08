# Fichier pour analyser un PDF et extraire le texte, les tableaux et les images, 
# en utilisant PyMuPDF pour lire le PDF et LlamaIndex pour créer des documents à partir du texte et des tableaux. 
# Les images raster sont extraites et enregistrées dans un répertoire spécifié, tandis que les figures vectorielles sont rendues en PNG et également enregistrées. Le résultat est encapsulé dans une classe ParsedPDF qui contient le nom du fichier, le hachage du fichier, le nombre de pages, les documents textuels et tabulaires, ainsi que les informations sur les images extraites.
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from llama_index.core import Document

from app.ingestion.figures import extract_vector_figures

MIN_IMAGE_SIZE = 80  # pixels : on ignore les petites icônes


# Alors nous avons la classe ParsedPDF qui est un conteneur pour les informations extraites d'un PDF. 
# Elle contient le nom du fichier, le hachage du fichier, le nombre de pages, une liste de documents textuels, une liste de documents tabulaires et une liste d'images. Les listes sont initialisées avec des valeurs par défaut vides à l'aide de field(default_factory=list) 
# Cette classe est utilisée pour encapsuler les résultats de l'analyse d'un PDF et faciliter l'accès aux différentes parties du contenu extrait
@dataclass
class ParsedPDF:
    filename: str
    file_hash: str
    page_count: int
    text_docs: list[Document] = field(default_factory=list)
    table_docs: list[Document] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)

# Alors nous avons la fonction compute_hash qui prend en entrée le chemin d'un fichier PDF et calcule son hachage SHA-256.
# Elle ouvre le fichier en mode binaire, lit le contenu par blocs de 1 Mo, et met à jour l'objet de hachage avec chaque bloc 
# Enfin, elle retourne le hachage sous forme de chaîne hexadécimale. Cette fonction est utilisée pour générer un identifiant unique pour le fichier PDF, ce qui permet de vérifier s'il a déjà été indexé dans la base de données.
def compute_hash(pdf_path: Path) -> str:
    sha = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()

# Alors nous avons la fonction parse_pdf qui prend en entrée le chemin d'un fichier PDF et un répertoire pour stocker les images extraites.
# Elle ouvre le PDF avec PyMuPDF, parcourt chaque page, extrait le texte et les tableaux, et enregistre les images raster et vectorielles dans le répertoire spécifié
def parse_pdf(pdf_path: str | Path, images_dir: str | Path) -> ParsedPDF:
    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    pdf = pymupdf.open(pdf_path)
    result = ParsedPDF(
        filename=pdf_path.name,
        file_hash=compute_hash(pdf_path),
        page_count=len(pdf),
    )
    seen_xrefs: set[int] = set()

    for page_index, page in enumerate(pdf):
        page_number = page_index + 1
        base_meta = {"source": pdf_path.name, "page": page_number}

        text = page.get_text("text").strip()
        if text:
            result.text_docs.append(
                Document(text=text, metadata={**base_meta, "type": "text"})
            )

        tables = page.find_tables().tables
        for t_index, table in enumerate(tables, start=1):
            markdown = table.to_markdown()
            if markdown.strip():
                result.table_docs.append(
                    Document(
                        text=markdown,
                        metadata={**base_meta, "type": "table", "table_index": t_index},
                    )
                )

        # Images raster (PNG/JPEG intégrés dans le PDF)
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            info = pdf.extract_image(xref)
            if info["width"] < MIN_IMAGE_SIZE or info["height"] < MIN_IMAGE_SIZE:
                continue

            filename = f"p{page_number}_img{xref}.{info['ext']}"
            path = images_dir / filename
            path.write_bytes(info["image"])
            result.images.append(
                {**base_meta, "type": "image", "file_path": str(path),
                 "width": info["width"], "height": info["height"]}
            )

        # Figures vectorielles (schémas dessinés), rendues en PNG
        table_rects = [pymupdf.Rect(t.bbox) for t in tables]
        result.images.extend(
            extract_vector_figures(page, page_number, images_dir, table_rects, base_meta)
        )

    pdf.close()
    return result
