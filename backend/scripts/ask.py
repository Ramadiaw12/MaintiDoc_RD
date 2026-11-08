import sys

from app.rag.pipeline import answer

result = answer(" ".join(sys.argv[1:]))

print("\n" + result["answer"])
print(f"\n[ancrée dans les sources : {result['grounded']} | citées : {result['cited']}]")
for s in result["sources"]:
    print(f"  [{s['n']}] page {s['page']} | {s['type']} | score {s['score']} | {s['extract'][:70]!r}")
