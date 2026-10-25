#
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