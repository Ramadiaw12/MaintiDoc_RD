# Fichier pour l'agent conversationnel. Il contient les fonctions pour construire les messages à envoyer au modèle de langage, exécuter l'agent avec les outils et le contexte, finaliser la réponse en gérant les citations et les sources, et fournir une réponse finale à partir d'une question et d'un historique de conversation. Il utilise le module `langchain` pour créer l'agent et le module `langchain_ollama` pour interagir avec le modèle Ollama.
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
AUTO_SEARCH = "search_documentation (automatique)"

AGENT_PROMPT = """Tu es MaintiDoc, assistant documentaire pour techniciens de maintenance.
Tu ne connais RIEN par toi-même : toute ta connaissance vient des extraits et des outils.

Règles strictes :
1. Réponds uniquement à partir des extraits fournis ou des résultats des outils.
2. Cite chaque affirmation avec son numéro, par exemple [1]. N'utilise jamais un numéro absent des résultats.
3. Si les extraits ne suffisent pas, appelle un outil : search_documentation (autre requête), search_figures (schémas, images) ou list_documents.
4. N'invente rien : aucune valeur, aucun code, aucune procédure.
5. Réponds dans la langue de la question, en une ou deux phrases."""

USER_TEMPLATE = """QUESTION : {question}

EXTRAITS DÉJÀ RETROUVÉS DANS LA DOCUMENTATION :
{context}

Si ces extraits contiennent la réponse, réponds directement en citant leurs numéros.
Sinon, appelle un outil pour chercher autre chose."""

# Là nous avons une fonction pour obtenir le modèle de chat Ollama avec mise en cache. Elle crée une instance de `ChatOllama` avec les paramètres de configuration et la retourne. L'utilisation de `@lru_cache` permet de ne créer qu'une seule instance du modèle pour toutes les requêtes, ce qui améliore les performances.
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

# ALors là nous avons une fonction pour construire les messages à envoyer au modèle de langage. Elle prend la question de l'utilisateur, l'historique des échanges et le contexte des extraits déjà retrouvés. Elle ne garde que les derniers tours de conversation, tronque les anciens messages et supprime les citations précédentes. Elle retourne une liste de messages formatés pour le modèle.
def build_messages(question: str, history: list[dict], context: str) -> list[dict]:
    """Gestion du contexte : derniers tours seulement, tronqués, sans anciennes citations."""
    messages = []
    for m in history[-2 * MAX_TURNS:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            content = re.sub(r"\[\d+\]", "", m["content"])[:MAX_HISTORY_CHARS].strip()
            messages.append({"role": m["role"], "content": content})
    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(question=question, context=context),
    })
    return messages

# Ici la fonction `run_agent` exécute l'agent conversationnel. Elle prend la question de l'utilisateur et l'historique des échanges. Elle construit les outils disponibles, récupère le contexte des extraits pertinents, crée l'agent avec le modèle de chat et les outils, et invoque l'agent avec les messages construits. Elle gère les appels d'outils supplémentaires et retourne le texte final, les sources utilisées et les outils appelés.
def run_agent(question: str, history: list[dict]) -> dict:
    registry: list[dict] = []
    tools = build_tools(registry)

    # Étape 1 : la recherche dans le RAG est TOUJOURS faite.
    search = next(t for t in tools if t.name == "search_documentation")
    context = search.invoke({"query": question})

    # Étape 2 : le LLM répond, ou appelle d'autres outils s'il en a besoin.
    agent = create_agent(model=get_chat_model(), tools=tools, system_prompt=AGENT_PROMPT)
    result = agent.invoke(
        {"messages": build_messages(question, history, context)},
        config={"recursion_limit": 8},  # coupe les boucles d'outils infinies
    )
    messages = result["messages"]
    extra_tools = [
        call["name"] for m in messages for call in (getattr(m, "tool_calls", None) or [])
    ]
    final = messages[-1].content
    text = final if isinstance(final, str) else str(final)
    return {
        "text": text.strip(),
        "sources": registry,
        "tools_used": [AUTO_SEARCH] + extra_tools,
    }


# Fonction pour finaliser la réponse de l'agent, en traitant les citations et les sources. Elle prend le texte généré par l'agent et la liste des sources utilisées. Elle extrait les numéros cités dans le texte, vérifie leur validité par rapport aux sources disponibles, retire les citations invalides et retourne un dictionnaire avec la réponse finale, si elle est sourcée, les citations valides et invalides.
def finalize(text: str, sources: list[dict]) -> dict:
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    valid = {s["n"] for s in sources}
    cited = sorted(n for n in numbers if n in valid)
    invalid = sorted(n for n in numbers if n not in valid)
    for n in invalid:
        text = text.replace(f"[{n}]", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return {"answer": text, "grounded": bool(cited), "cited": cited, "invalid_citations": invalid}

# Fonction principale pour obtenir la réponse de l'agent. Elle prend la question de l'utilisateur et l'historique des échanges. Elle exécute l'agent avec la question et l'historique, gère les cas où l'agent échoue ou ne produit pas de réponse, et retourne un dictionnaire avec la réponse finale, si elle est sourcée, les citations valides et invalides, les sources consultées, le mode utilisé (agent ou RAG fallback) et les outils appelés.
def agent_answer(question: str, history: list[dict] | None = None) -> dict:
    history = history or []
    try:
        run = run_agent(question, history)
    except Exception as error:
        fallback = rag_answer(question)
        return {**fallback, "mode": "rag_fallback", "tools_used": [],
                "fallback_reason": f"L'agent a échoué : {error}"}

    if not run["text"]:
        fallback = rag_answer(question)
        return {**fallback, "mode": "rag_fallback", "tools_used": run["tools_used"],
                "fallback_reason": "L'agent n'a produit aucune réponse."}

    if not run["sources"] and "list_documents" not in run["tools_used"]:
        return {"answer": NOT_FOUND, "grounded": False, "cited": [], "invalid_citations": [],
                "sources": [], "mode": "agent", "tools_used": run["tools_used"]}

    return {**finalize(run["text"], run["sources"]), "sources": run["sources"],
            "mode": "agent", "tools_used": run["tools_used"]}