# Ce script est un exemple d'utilisation de la fonction parse_pdf pour analyser un fichier PDF
# qui est stocké dans le répertoire data/samples. Il affiche ensuite les informations extraites du PDF
from pathlib import Path

from app.ingestion.parser import parse_pdf

# Définir le chemin du fichier PDF à analyser
BASE = Path(__file__).resolve().parents[1]
pdf = BASE / "data" / "samples" / "attention_is_all_you_need.pdf"

# Passer le chemin du fichier PDF à la fonction parse_pdf
result = parse_pdf(pdf, BASE / "data" / "extracted" / "test")

print("Fichier  :", result.filename)
print("Hash     :", result.file_hash[:12], "...")
print("Pages    :", result.page_count)
print("Textes   :", len(result.text_docs))
print("Tableaux :", len(result.table_docs))
print("Images   :", len(result.images))

print("\n--- Début de la page 1 ---")
print(result.text_docs[0].text[:400])

# Afficher le premier tableau extrait, s'il existe
if result.table_docs:
    t = result.table_docs[0]
    print(f"\n--- Premier tableau (page {t.metadata['page']}) ---")
    print(t.text[:500])