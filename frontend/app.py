import os
from pathlib import Path

import httpx
import streamlit as st

BASE_URL = os.getenv("MAINTIDOC_API_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"

st.set_page_config(page_title="MaintiDoc", page_icon="🔧", layout="wide")


def api_ok() -> bool:
    try:
        httpx.get(f"{BASE_URL}/health", timeout=3).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text


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


# ---------- Barre latérale : état, envoi de PDF, documents ----------
connected = api_ok()
with st.sidebar:
    st.title("🔧 MaintiDoc")
    if connected:
        st.success("API connectée")
    else:
        st.error("API injoignable. Dans backend/ : uv run uvicorn app.main:app --port 8000")

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
                else:
                    st.error(error_message(r))
            except httpx.HTTPError as error:
                st.error(f"Erreur de connexion : {error}")

    st.subheader("Documents indexés")
    if connected:
        try:
            documents = httpx.get(f"{API}/documents", timeout=10).json()
        except (httpx.HTTPError, ValueError):
            documents = []
        if not documents:
            st.caption("Aucun document.")
        for d in documents:
            left, right = st.columns([5, 1])
            left.write(f"📄 {d['filename']} ({d['page_count']} p.)")
            if right.button("🗑", key=f"delete_{d['id']}", help="Supprimer ce document"):
                httpx.delete(f"{API}/documents/{d['id']}", timeout=60)
                st.rerun()

    st.divider()
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------- Zone principale : conversation ----------
st.header("Pose ta question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message["payload"])

question = st.chat_input("Ex. : How many layers does the encoder have?", disabled=not connected)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la documentation..."):
            try:
                r = httpx.post(f"{API}/chat", json={"question": question}, timeout=300)
                if r.status_code == 200:
                    payload = r.json()
                    render_assistant(payload)
                    st.session_state.messages.append({"role": "assistant", "payload": payload})
                else:
                    st.error(error_message(r))
            except httpx.HTTPError as error:
                st.error(f"Erreur de connexion : {error}")
