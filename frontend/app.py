import os
import time
from pathlib import Path

import httpx
import streamlit as st

BASE_URL = os.getenv("MAINTIDOC_API_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"

EXAMPLES = [
    "How many layers does the encoder have?",
    "What is multi-head attention?",
    "What BLEU score did the big model reach on English-to-German?",
]

st.set_page_config(page_title="MaintiDoc", page_icon="🔧", layout="wide")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("confirm_delete", None)


# ---------- Appels à l'API ----------
@st.cache_data(ttl=5, show_spinner=False)
def api_ok() -> bool:
    try:
        httpx.get(f"{BASE_URL}/health", timeout=3).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def list_documents() -> list[dict]:
    try:
        r = httpx.get(f"{API}/documents", timeout=10)
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError):
        return []


def error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text
    detail = data.get("detail", response.text) if isinstance(data, dict) else response.text
    if isinstance(detail, list):  # erreurs de validation (422)
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def ask_api(question: str, top_k: int, use_agent: bool, history: list[dict]) -> tuple[dict | None, str | None]:
    try:
        if use_agent:
            turns = [
                {"role": m["role"],
                 "content": m["content"] if m["role"] == "user" else m["payload"]["answer"]}
                for m in history
                if m["role"] == "user" or "payload" in m
            ]
            r = httpx.post(
                f"{API}/agent/chat", json={"question": question, "history": turns[-6:]}, timeout=600
            )
        else:
            r = httpx.post(
                f"{API}/chat", json={"question": question, "top_k": top_k}, timeout=300
            )
    except httpx.HTTPError as error:
        return None, f"Erreur de connexion : {error}"
    if r.status_code != 200:
        return None, error_message(r)
    return r.json(), None


# ---------- Affichage ----------
def render_assistant(payload: dict) -> None:
    st.markdown(payload["answer"])
    if payload["sources"] and not payload["grounded"]:
        st.warning("Aucune source vérifiable n'est citée : vérifie dans le document avant d'agir.")
    if payload["sources"]:
        with st.expander(f"Sources consultées ({len(payload['sources'])})"):
            for s in payload["sources"]:
                cited = " · ✅ citée" if s["n"] in payload["cited"] else ""
                st.markdown(f"**[{s['n']}]** page {s['page']} · {s['type']} · score {s['score']}{cited}")
                st.caption(s["extract"])
                path = s.get("file_path")
                if s["type"] == "image" and path and Path(path).exists():
                    st.image(path, width=400)
    if payload.get("elapsed"):
        st.caption(f"Réponse en {payload['elapsed']:.0f} s")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif "error" in message:
            st.error(message["error"])
        else:
            render_assistant(message["payload"])


# ---------- Barre latérale ----------
connected = api_ok()
documents = list_documents() if connected else []

with st.sidebar:
    st.title("🔧 MaintiDoc")
    if connected:
        st.success("API connectée")
    else:
        st.error("API injoignable. Dans backend/ : uv run uvicorn app.main:app --port 8000")
    st.caption(f"API : {BASE_URL}")

    st.subheader("Réglages")
    top_k = st.slider("Passages consultés", 1, 8, 3,
                      help="Plus de passages donne plus de contexte, mais des réponses plus lentes.")

    st.subheader("Ajouter un PDF")
    uploaded = st.file_uploader("Choisis un PDF", type=["pdf"])
    with_images = st.checkbox("Décrire les images (lent)", value=False)
    if st.button("Indexer", disabled=not (connected and uploaded)):
        with st.spinner("Indexation en cours, cela peut prendre plusieurs minutes..."):
            try:
                r = httpx.post(
                    f"{API}/documents",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    params={"with_images": with_images},
                    timeout=1800,
                )
                if r.status_code == 200:
                    result = r.json()
                    if result["status"] == "already_indexed":
                        st.info("Ce PDF est déjà indexé.")
                    else:
                        st.success(f"Indexé : {result.get('chunks')} passages.")
                    list_documents.clear()
                    documents = list_documents()
                else:
                    st.error(error_message(r))
            except httpx.HTTPError as error:
                st.error(f"Erreur de connexion : {error}")

    st.subheader("Documents indexés")
    if not documents:
        st.caption("Aucun document.")
    for d in documents:
        left, right = st.columns([5, 1])
        left.write(f"📄 {d['filename']} ({d['page_count']} p.)")
        if right.button("🗑", key=f"delete_{d['id']}", help="Supprimer ce document"):
            st.session_state.confirm_delete = d["id"]

    if st.session_state.confirm_delete is not None:
        st.warning("Supprimer ce document et tous ses passages ? Cette action est définitive.")
        yes, no = st.columns(2)
        if yes.button("Oui, supprimer"):
            r = httpx.delete(f"{API}/documents/{st.session_state.confirm_delete}", timeout=60)
            if r.status_code not in (200, 204):
                st.error(error_message(r))
            st.session_state.confirm_delete = None
            list_documents.clear()
            st.rerun()
        if no.button("Annuler"):
            st.session_state.confirm_delete = None
            st.rerun()

    st.divider()
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------- Zone principale ----------
st.header("Pose ta question")
st.caption("Les réponses reposent uniquement sur les documents indexés, avec leurs sources.")

if connected and not documents:
    st.info("Aucun document indexé : ajoute un PDF dans la barre latérale pour commencer.")

typed = st.chat_input(
    "Ex. : How many layers does the encoder have?",
    disabled=not (connected and documents),
)
question = typed or st.session_state.pending
st.session_state.pending = None

for message in st.session_state.messages:
    render_message(message)

if not st.session_state.messages and not question and connected and documents:
    st.write("Essaie une question :")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la documentation..."):
            started = time.perf_counter()
            payload, error = ask_api(question, top_k)
        if error:
            st.error(error)
            reply = {"role": "assistant", "error": error}
        else:
            payload["elapsed"] = time.perf_counter() - started
            render_assistant(payload)
            reply = {"role": "assistant", "payload": payload}
    st.session_state.messages.append(reply)