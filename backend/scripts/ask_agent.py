# Ce script permet d'interroger l'agent en ligne de commande. Il prend la question en argument, appelle la fonction agent_answer pour obtenir la réponse, et affiche le résultat avec les informations sur le mode utilisé, les outils utilisés, les citations et les sources consultées
import sys

from app.agent.agent import agent_answer

# On appelle la fonction agent_answer avec la question fournie en argument de la ligne de commande. La question est construite en joignant tous les arguments passés après le nom du script. La fonction retourne un dictionnaire contenant la réponse de l'agent, le mode utilisé (agent ou RAG fallback), les outils utilisés, les citations valides et invalides, et les sources consultées
result = agent_answer(" ".join(sys.argv[1:]))

# On affiche la réponse de l'agent, le mode utilisé, les outils utilisés et les citations valides. 
# Si un repli a été effectué (fallback), on affiche également la raison du repli. Enfin, on parcourt les sources consultées et on affiche leurs informations (numéro, page, type, score et extrait)
print("\n" + result["answer"])
print(f"\n[mode : {result['mode']} | outils : {result['tools_used']} | citées : {result['cited']}]")
if result.get("fallback_reason"):
    print("Repli :", result["fallback_reason"])
for s in result["sources"]:
    print(f"  [{s['n']}] page {s['page']} | {s['type']} | score {s['score']} | {s['extract'][:60]!r}")
