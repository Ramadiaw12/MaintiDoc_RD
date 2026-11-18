# Ce file nous permet de créer un agent LangChain pour répondre à des questions techniques à partir de documents PDF indexés. L'agent utilise des outils pour rechercher dans la documentation et les figures, et il est configuré pour suivre des règles strictes afin de fournir des réponses précises et sourcées.
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

AGENT_PROMPT = AGENT_PROMPT = """Tu es MaintiDoc, assistant documentaire pour techniciens de maintenance.
Tu ne connais RIEN par toi-même : toute ta connaissance vient des outils.

Procédure obligatoire :
1. Pour toute question technique, appelle d'abord search_documentation. Ne réponds jamais sans avoir appelé un outil.
2. Pour voir un schéma ou une image, appelle search_figures.
3. Pour savoir quels documents existent, appelle list_documents.
4. Rédige ensuite ta réponse uniquement avec les résultats reçus, en citant chaque affirmation avec son numéro, par exemple [1].
5. N'invente rien : aucune valeur, aucun code, aucune procédure.
6. Réponds dans la langue de la question, en une ou deux phrases."""

# Alors là c'est un peu technique : on crée un modèle de chat Ollama avec des paramètres spécifiques pour l'agent. On utilise `lru_cache` pour ne pas recréer le modèle à chaque fois, ce qui économise du temps et des ressources
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

# Alors ici on définit une fonction pour construire les messages à envoyer à l'agent, en prenant en compte l'historique des échanges. On ne garde que les derniers tours et on tronque les anciens messages pour ne pas dépasser la limite de caractères. On enlève aussi les citations anciennes pour éviter la confusion.
def build_messages(question: str, history: list[dict]) -> list[dict]:
    """Gestion du contexte : derniers tours seulement, tronqués, sans anciennes citations."""
    messages = []
    for m in history[-2 * MAX_TURNS:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            content = re.sub(r"\[\d+\]", "", m["content"])[:MAX_HISTORY_CHARS].strip()
            messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": question})
    return messages

# Fonction principale pour exécuter l'agent
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

# Ici on définit une fonction pour finaliser la réponse de l'agent. Elle vérifie les citations dans le texte, garde seulement celles qui sont valides et supprime les citations invalides. Elle retourne un dictionnaire avec la réponse finale, si elle est sourcée, les citations valides et invalides
def finalize(text: str, sources: list[dict]) -> dict:
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    valid = {s["n"] for s in sources}
    cited = sorted(n for n in numbers if n in valid)
    invalid = sorted(n for n in numbers if n not in valid)
    for n in invalid:
        text = text.replace(f"[{n}]", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return {"answer": text, "grounded": bool(cited), "cited": cited, "invalid_citations": invalid}

# Fonction principale pour obtenir la réponse de l'agent. Elle essaie d'exécuter l'agent avec la question et l'historique. Si l'agent échoue ou n'utilise aucun outil, elle tombe en mode "fallback" et utilise le pipeline RAG pour répondre. Elle retourne un dictionnaire avec la réponse, les sources, le mode utilisé et les outils utilisés
def agent_answer(question: str, history: list[dict] | None = None) -> dict:
    history = history or []
    try:
        run = run_agent(question, history)
    except Exception as error:
        fallback = rag_answer(question)
        return {**fallback, "mode": "rag_fallback", "tools_used": [],
                "fallback_reason": f"L'agent a échoué : {error}"}
    # Si l'agent n'a utilisé aucun outil, on tombe en mode "fallback" et on utilise le pipeline RAG pour répondre. Cela permet de garantir que l'utilisateur obtienne une réponse même si l'agent ne peut pas traiter la question correctement
    if not run["tools_used"]:
        fallback = rag_answer(question)
        return {**fallback, "mode": "rag_fallback", "tools_used": [],
                "fallback_reason": "L'agent n'a utilisé aucun outil."}

    # Si l'agent n'a trouvé aucune source pertinente et qu'il n'a pas utilisé l'outil `list_documents`, on retourne un message indiquant qu'aucune réponse n'a été trouvée. Cela permet d'informer l'utilisateur que la question n'a pas pu être résolue avec les documents disponibles
    if not run["sources"] and "list_documents" not in run["tools_used"]:
        return {"answer": NOT_FOUND, "grounded": False, "cited": [], "invalid_citations": [],
                "sources": [], "mode": "agent", "tools_used": run["tools_used"]}

    # Sinon, on finalise la réponse de l'agent en vérifiant les citations et en nettoyant le texte. On retourne un dictionnaire avec la réponse finale, les sources, le mode utilisé et les outils utilisés
    return {**finalize(run["text"], run["sources"]), "sources": run["sources"],
            "mode": "agent", "tools_used": run["tools_used"]}
