import sys

from app.agent.agent import agent_answer

result = agent_answer(" ".join(sys.argv[1:]))

print("\n" + result["answer"])
print(f"\n[mode : {result['mode']} | outils : {result['tools_used']} | citées : {result['cited']}]")
if result.get("fallback_reason"):
    print("Repli :", result["fallback_reason"])
for s in result["sources"]:
    print(f"  [{s['n']}] page {s['page']} | {s['type']} | score {s['score']} | {s['extract'][:60]!r}")
