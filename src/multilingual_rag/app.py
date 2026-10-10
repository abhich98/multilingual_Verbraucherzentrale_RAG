"""Streamlit interface for the multilingual Verbraucherzentrale RAG assistant."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer # noqa
import faiss  # noqa

from multilingual_rag.config import AppConfig, ConfigurationError
from multilingual_rag.providers.openrouter import OpenRouterProvider
from multilingual_rag.rag.pipeline import answer_question
from multilingual_rag.retrieval.embeddings import E5Embedder
from multilingual_rag.retrieval.faiss_store import SearchResult, search_index


logger = logging.getLogger(__name__)
load_dotenv()


@st.cache_resource(show_spinner="Lade das lokale Suchmodell …")
def _get_embedder(model_name: str, revision: str | None, cache_dir: str | None) -> E5Embedder:
    logger.info("Initializing E5 embedder with model: %s, revision: %s", model_name, revision)
    return E5Embedder(model_name, device="cpu", revision=revision, cache_dir=cache_dir)


def _configured_api_key() -> str | None:
    try:
        secret = st.secrets.get("OPENROUTER_API_KEY")
    except FileNotFoundError:
        secret = None
    return str(secret) if secret else None


def _render_sources(sources: list[SearchResult]) -> None:
    if not sources:
        return
    with st.expander(f"Quellen ({len(sources)})"):
        for index, result in enumerate(sources, start=1):
            chunk = result.chunk
            title = str(chunk.get("title") or chunk.get("source_path") or "Quelle")
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end", page_start)
            page_text = f" · Seite {page_start}"
            if page_end and page_end != page_start:
                page_text += f"–{page_end}"
            st.markdown(f"**[S{index}] {title}**{page_text}")
            url = chunk.get("url")
            if url:
                st.link_button("Artikel öffnen", str(url), icon="↗")
            with st.container(border=True):
                st.write(str(chunk.get("text", "")))


def _render_history(messages: list[dict[str, Any]]) -> None:
    for item in messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                _render_sources(item.get("sources", []))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    st.set_page_config(page_title="Verbraucher-Ratgeber", page_icon="V", layout="centered")
    st.title("Verbraucher-Ratgeber")
    st.caption("Antworten auf Basis der Verbraucherzentrale-Artikel in Ihrem lokalen Index.")

    api_key_override = _configured_api_key()
    try:
        config = AppConfig.from_environment(api_key_override=api_key_override)
    except ConfigurationError as exc:
        st.info(
            "Die Anwendung ist noch nicht konfiguriert. Setzen Sie "
            "`OPENROUTER_API_KEY` und `OPENROUTER_MODEL` als Umgebungsvariablen "
            "oder hinterlegen Sie den API-Schlüssel in Streamlit Secrets."
        )
        st.code("OPENROUTER_MODEL=provider/model-name\nOPENROUTER_API_KEY=…", language="text")
        st.caption(str(exc))
        return

    pointer_path = config.index_root / "current.json"
    if not pointer_path.is_file():
        st.error(f"Kein aktiver Suchindex gefunden unter `{config.index_root}`.")
        st.code("uv run rag-ingest\nuv run rag-index", language="bash")
        return

    with st.sidebar:
        st.subheader("Suche")
        top_k = st.slider("Quellen pro Antwort", min_value=1, max_value=10, value=min(config.top_k, 10))
        st.caption(f"Modell: `{config.openrouter_model}`")
        if st.button("Chat leeren", icon="🗑", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()

    messages: list[dict[str, Any]] = st.session_state.setdefault("chat_messages", [])
    _render_history(messages)

    question = st.chat_input("Stellen Sie eine Frage zu Verbraucher-Themen …")
    logger.info("User asked question: %s", question)
    if not question:
        return

    recent_history = [
        {"role": message["role"], "content": message["content"]}
        for message in messages[-8:]
    ]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Suche in den Artikeln und formuliere eine Antwort …"):
            embedder = _get_embedder(config.embedding_model, config.embedding_revision, config.embedding_cache_dir)
            logger.info("Initializing retriever with embedder")
            retriever = lambda query, *, top_k: search_index(
                config.index_root,
                query,
                top_k=top_k,
                model_name=config.embedding_model,
                revision=config.embedding_revision,
                embedder=embedder,
            )
            logger.info("OpenRouter provider initialization")
            provider = OpenRouterProvider(
                config.openrouter_api_key,
                config.openrouter_model,
                endpoint=config.openrouter_url,
                timeout_seconds=config.request_timeout_seconds,
            )
            logger.info("Answering question with retriever and provider")
            response = answer_question(
                question,
                retrieve=retriever,
                provider=provider,
                top_k=top_k,
                max_context_characters=config.max_context_characters,
                history=recent_history,
            )
    except Exception as exc:
        logger.exception("RAG turn failed")
        messages.pop()
        st.error(str(exc))
        return

    messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        }
    )
    with st.chat_message("assistant"):
        st.markdown(response.answer)
        _render_sources(response.sources)


if __name__ == "__main__":
    main()