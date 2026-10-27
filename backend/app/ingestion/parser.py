# Configution de l'application en utilisant Pydantic pour gérer les paramètres de configuration. 
# Les paramètres sont chargés à partir d'un fichier .env situé à la racine du projet
import hashlib 
from dataclasses import dataclass, field
from pathlib import Path # Manipule les chemins des fichiers proprement 

import pymupdf
from llama_index.core import Document

# Crée une classe qui sert uniquement à regrouper les données extraites d'un PDF, telles que le nom du fichier, le hachage du fichier, le nombre de pages, les documents textuels et tabulaires, ainsi que les images. Cette classe est utilisée pour stocker et organiser les informations extraites d'un PDF de manière structurée.
@dataclass
class ParsedPDF:
    filename: str
    file_hash: str
    page_count: int
    # Document est l’objet LlamaIndex qui contient un text et un dictionnaire metadata
    text_docs: list[Document] = field(default_factory=list)
    table_docs: list[Document] = field(default_factory=list)
    images: list[dict] = field(default_factory=list) # Commence par une liste vide pour stocker les images extraites du PDF. Chaque image est représentée par un dictionnaire contenant des informations sur l'image, telles que son nom de fichier et son contenu binaire

    # Le hash du fichier PDF est calculé en lisant le fichier par blocs de 1 Mo et en mettant à jour l'objet de hachage SHA-256 avec chaque bloc. Le hachage final est renvoyé sous forme de chaîne hexadécimale.
    # Le hash est l’empreinte unique d’un fichier. Deux PDF identiques donnent le même hash, ce qui permettra d’éviter de ré-indexer un document déjà traité
    def compute_hash(pdf_path: Path) -> str:
        sha = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(block)
        return sha.hexdigest() # Renvoie le hachage final sous forme de chaîne hexadécimale, qui peut être utilisé pour identifier de manière unique le fichier PDF.


    MIN_IMAGE_SIZE = 80  # pixels : on ignore les petites icônes

    # La fonction parse_pdf lit un fichier PDF, crée un répertoire pour stocker les images extraites, et retourne un objet ParsedPDF contenant des informations sur le PDF, telles que le nom du fichier, le hachage du fichier, le nombre de pages, ainsi que les documents textuels et tabulaires et les images extraites
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

        # Parcourt chaque page du PDF, extrait le texte et les tableaux, et les stocke dans l'objet ParsedPDF. Pour chaque page, il crée un dictionnaire de métadonnées contenant le nom du fichier source et le numéro de page. Si du texte est trouvé sur la page, il est ajouté à la liste text_docs de l'objet ParsedPDF avec les métadonnées correspondantes. De même, si des tableaux sont trouvés sur la page, ils sont convertis en format Markdown et ajoutés à la liste table_docs avec les métadonnées correspondantes, y compris l'index du tableau.
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            base_meta = {"source": pdf_path.name, "page": page_number}

            text = page.get_text("text").strip()
            if text:
                result.text_docs.append(
                    Document(text=text, metadata={**base_meta, "type": "text"})
                )

            # Parcourt chaque tableau trouvé sur la page, le convertit en format Markdown, et l'ajoute à la liste table_docs de l'objet ParsedPDF avec les métadonnées correspondantes, y compris l'index du tableau. Si le tableau est vide (après suppression des espaces), il n'est pas ajouté à la liste.
            for t_index, table in enumerate(page.find_tables().tables, start=1):
                markdown = table.to_markdown()
                if markdown.strip():
                    result.table_docs.append(
                        Document(
                            text=markdown,
                            metadata={**base_meta, "type": "table", "table_index": t_index},
                        )
                    ) 
            # Parcourt chaque image trouvée sur la page, vérifie si elle a déjà été vue (en utilisant son xref), et si elle est suffisamment grande (en vérifiant sa largeur et sa hauteur). Si l'image est valide, elle est extraite, enregistrée dans le répertoire des images avec un nom de fichier unique basé sur le numéro de page et le xref, et ajoutée à la liste images de l'objet ParsedPDF avec les métadonnées correspondantes, y compris le chemin du fichier, la largeur et la hauteur de l'image.
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                info = pdf.extract_image(xref)
                if info["width"] < MIN_IMAGE_SIZE or info["height"] < MIN_IMAGE_SIZE:
                    continue
                # Crée un nom de fichier unique pour l'image en utilisant le numéro de page et le xref, et en ajoutant l'extension appropriée. Le nom de fichier est ensuite utilisé pour créer un chemin complet vers le répertoire des images, et l'image est enregistrée sur le disque à cet emplacement
                filename = f"p{page_number}_img{xref}.{info['ext']}"
                path = images_dir / filename
                path.write_bytes(info["image"])

                # Ajout des métadonnées de l'image à la liste images de l'objet ParsedPDF, y compris le type d'image, le chemin du fichier, la largeur et la hauteur de l'image. Les métadonnées sont combinées avec les métadonnées de base (nom du fichier source et numéro de page) pour fournir des informations complètes sur l'image extraite
                result.images.append(
                    {**base_meta, "type": "image", "file_path": str(path),
                    "width": info["width"], "height": info["height"]}
                )

        pdf.close()
        return result