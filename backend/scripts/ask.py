# Ce script permet d'interroger le modèle RAG en ligne de commande. Il prend la question en argument, appelle la fonction answer pour obtenir la réponse, et affiche le résultat avec les informations sur les citations et les sources consultées
import sys

from app.rag.pipeline import answer

result = answer(" ".join(sys.argv[1:]))

print("\n" + result["answer"])
print(f"\n[ancrée : {result['grounded']} | citées : {result['cited']} | invalides retirées : {result['invalid_citations']}]")
for s in result["sources"]:
    print(f"  [{s['n']}] page {s['page']} | {s['type']} | score {s['score']} | {s['extract'][:70]!r}")
