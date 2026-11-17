import re
from functools import lru_cache

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from app.agent.tools import build_tools
from app.core.config import settings
from app.rag.llm import MAX_ANSWER_TOKENS, NUM_CTX
from app.rag.pipeline import NOT_FOUND
from app.rag.pipeline import answer as rag_answer

MAX_TURNS = 3            # nombre d'échanges gardés en mémoire
MAX_HISTORY_CHARS = 600  # taille maximale d'un ancien message

AGENT_PROMPT = """Tu es MaintiDoc, un assistant documentaire pour techniciens de maintenance.

Outils :
- search_documentation : pour toute question technique.
- search_figures : pour voir un schéma, un diagramme ou une image.
- list_documents : pour savoir quels documents sont disponibles.

Règles strictes :
1. Pour toute question technique, appelle d'abord un outil de recherche.
2. Réponds UNIQUEMENT à partir des résultats des outils.
3. Cite chaque affirmation avec le numéro de sa source, par exemple [1]. N'utilise jamais un numéro absent des résultats.
4. Si les résultats indiquent qu'aucun passage n'est pertinent, réponds : "Je ne trouve pas cette information dans la documentation."
5. N'invente jamais de valeur, de code ni de procédure.
6. Réponds dans la langue de la question, de façon courte et précise."""


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_ctx=NUM_CTX,
        num_predict=MAX_ANSWER_TOKENS,
        keep_alive="30m",
    )


def build_messages(question: str, history: list[dict]) -> list[dict]:
    """Gestion du contexte : derniers tours seulement, tronqués, sans anciennes citations."""
    messages = []
    for m in history[-2 * MAX_TURNS:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            content = re.sub(r"\[\d+\]", "", m["content"])[:MAX_HISTORY_CHARS].strip()
            messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def run_agent(question: str, history: list[dict]) -> dict:
    registry: list[dict] = []
    agent = create_agent(
        model=get_chat_model(),
        tools=build_tools(registry),
        system_prompt=AGENT_PROMPT,
    )
    result = agent.invoke(
        {"messages": build_messages(question, history)},
        config={"recursion_limit": 8},  # coupe les boucles d'outils infinies
    )
    messages = result["messages"]
    tools_used = [
        call["name"] for m in messages for call in (getattr(m, "tool_calls", None) or [])
    ]
    final = messages[-1].content
    text = final if isinstance(final, str) else str(final)
    return {"text": text.strip(), "sources": registry, "tools_used": tools_used}


def finalize(text: str, sources: list[dict]) -> dict:
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    valid = {s["n"] for s in sources}
    cited = sorted(n for n in numbers if n in valid)
    invalid = sorted(n for n in numbers if n not in valid)
    for n in invalid:
        text = text.replace(f"[{n}]", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return {"answer": text, "grounded": bool(cited), "cited": cited, "invalid_citations": invalid}


def agent_answer(question: str, history: list[dict] | None = None) -> dict:
    history = history or []
    try:
        run = run_agent(question, history)
    except Exception as error:
        fallback = rag_answer(question)
        return {**fallback, "mode": "rag_fallback", "tools_used": [],
                "fallback_reason": f"L'agent a échoué : {error}"}

    if not run["tools_used"]:
        fallback = rag_answer(question)
        return {**fallback, "mode": "rag_fallback", "tools_used": [],
                "fallback_reason": "L'agent n'a utilisé aucun outil."}

    if not run["sources"] and "list_documents" not in run["tools_used"]:
        return {"answer": NOT_FOUND, "grounded": False, "cited": [], "invalid_citations": [],
                "sources": [], "mode": "agent", "tools_used": run["tools_used"]}

    return {**finalize(run["text"], run["sources"]), "sources": run["sources"],
            "mode": "agent", "tools_used": run["tools_used"]}
