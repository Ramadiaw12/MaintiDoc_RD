from pathlib import Path

from app.ingestion.parser import parse_pdf

BASE = Path(__file__).resolve().parents[1]
pdf = BASE / "data" / "samples" / "attention_is_all_you_need.pdf"

result = parse_pdf(pdf, BASE / "data" / "extracted" / "test")

print("Fichier  :", result.filename)
print("Hash     :", result.file_hash[:12], "...")
print("Pages    :", result.page_count)
print("Textes   :", len(result.text_docs))
print("Tableaux :", len(result.table_docs))
print("Images   :", len(result.images))

print("\n--- Début de la page 1 ---")
print(result.text_docs[0].text[:400])

if result.table_docs:
    t = result.table_docs[0]
    print(f"\n--- Premier tableau (page {t.metadata['page']}) ---")
    print(t.text[:500])