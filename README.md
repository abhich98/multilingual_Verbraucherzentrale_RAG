

# Multilingual Verbraucherzentrale RAG

A local retrieval-augmented generation application for answering consumer-protection questions from German Verbraucherzentrale articles. The project combines domain-specific PDF processing with a reproducible multilingual search and question-answering stack.

## Project Scope

### Domain scope

This project focuses on consumer-protection information published in Verbraucherzentrale articles (downloaded as PDFs). The source documents and their metadata are in German, and the application answers questions about topics covered by the indexed articles while providing links and page references back to the source material. Because the embedding model is multilingual, English questions can also be used for cross-language retrieval, although this repository does not contain an English source corpus.

### Technical scope

- DVC-managed PDF corpus.
- PyMuPDF extraction with publisher-specific boilerplate filtering and URL extraction.
- Page-local text chunks with provenance metadata written as UTF-8 JSONL.
- Local `intfloat/multilingual-e5-small` passage and query embeddings.
- Versioned FAISS indexes with cosine-similarity retrieval.
- OpenRouter-backed answer generation with retrieved context and citations.
- Streamlit chat interface for local use.

The implemented workflow is:

```text
DVC PDF corpus
		-> PDF extraction and filtering
		-> JSONL chunks and manifest
		-> E5 embeddings
		-> versioned FAISS index
		-> retrieved context and cited answer
```

## Requirements

- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/)
- Git and DVC
- Network access to download the Hugging Face embedding model
- An OpenRouter API key and model identifier for interactive answers

Install the project from the repository root:

```bash
uv sync
```

The first indexing run downloads the embedding model and creates local FAISS artifacts. The model cache and index can require substantial disk space. The documented workflow uses the CPU device by default.

## Data and Indexing Workflow

### Source data

Around 65 PDF articles are included in the initial corpus. The Git-tracked file `data/Verbraucherzentrale_artikel.dvc` is a pointer to the raw PDF corpus. The PDF files themselves are managed by DVC and should not be committed to Git.

### Ingesting the PDFs

The PDFs stored in the `data/Verbraucherzentrale_artikel/` directory are ingested by default when running the ingestion command. The relevant text is chunked into page-local segments with provenance metadata. Run the default ingestion command:

```bash
uv run rag-ingest
```

To customize the run:

```bash
uv run rag-ingest \
	--input data/Verbraucherzentrale_artikel \
	--output artifacts/chunks.jsonl \
	--chunk-size 1000 \
	--overlap 150
```

Chunks are kept within their source page. The default maximum chunk size is 1,000 characters with 150 characters of overlap. The ingestion pipeline normalizes extracted text, removes known Verbraucherzentrale print and article boilerplate, extracts article URLs when possible, and preserves source metadata. The chunks are saved in the `artifacts/` directory.

### Building the retrieval index

The retrieval index is built from the ingested chunks and is used to quickly find relevant context for user queries. It is based on FAISS and uses the E5 embeddings for semantic search. 
Build the default local index after ingestion:

```bash
uv run rag-index
```

The default command reads `artifacts/chunks.jsonl`, writes under `artifacts/indexes`, uses `intfloat/multilingual-e5-small`, runs on CPU, and uses a batch size of 32. This model is picked for its support of English and German languages and considering the small size of the initial corpus.

Available options include:

```bash
uv run rag-index \
	--chunks artifacts/chunks.jsonl \
	--index-root artifacts/indexes \
	--model intfloat/multilingual-e5-small \
	--revision REVISION \
	--device cpu \
	--cache-dir .cache/ \
	--batch-size 32
```

E5 uses the `passage: ` prefix for indexed chunks and the `query: ` prefix for user questions. Embeddings are normalized and searched with inner product, which is equivalent to cosine similarity for these vectors. `current.json` identifies the active complete build. If the embedding model or revision changes, rebuild the index before starting the application. Retrieval rejects an active index whose embedding metadata does not match the configured model or revision.

**Retrieval is available through the Python API and the Streamlit application.**

## Evaluation and metrics

With the current RAG setup, preliminary **retrieval** evaluation shows decent results, indicating that the system is able to find relevant context for user queries. But, there is still room for improvement, there appears to be a gap in the quality of retrieved content for German vs English queries. Since, the corpus is primarily in German, the system appears to perform better for German queries compared to English ones.


| Group | n (# queries) | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 32 | 0.653 | 0.385 | 0.586 | 0.690 | 0.906 | 0.531 | 0.719 | 0.812 | 1.000 |
| German (`de`) | 11 | 0.753 | 0.545 | **0.758** | **0.758** | 1.000 | 0.636 | 0.909 | 0.909 | 1.000 |
| English (`en`) | 21 | 0.601 | 0.302 | **0.496** | **0.655** | 0.857 | 0.476 | 0.619 | 0.762 | 1.000 |

**Future directions**: Change the chunk size to see if that improves the retrieval performance for English queries.

## Configuration and Application

The application requires an OpenRouter API key and a model identifier. Create a local `.env` file, or export the variables in your shell:

```dotenv
OPENROUTER_API_KEY=replace-with-your-key
OPENROUTER_MODEL=provider/model-name
```

Never commit `.env` or API credentials.

### Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes | None | OpenRouter authentication key |
| `OPENROUTER_MODEL` | Yes | None | OpenRouter model in `provider/model` form |
| `OPENROUTER_URL` | No | `https://openrouter.ai/api/v1/chat/completions` | OpenRouter endpoint |
| `OPENROUTER_TIMEOUT_SECONDS` | No | `45` | Generation request timeout |
| `RAG_INDEX_ROOT` | No | `artifacts/indexes` | Active FAISS index directory |
| `EMBEDDING_MODEL` | No | `intfloat/multilingual-e5-small` | Sentence Transformers model |
| `EMBEDDING_REVISION` | No | Unset | Optional model revision |
| `EMBEDDING_CACHE_DIR` | No | `.cache/` | Local model cache directory |
| `RAG_TOP_K` | No | `5` | Default number of retrieved sources |
| `RAG_MAX_CONTEXT_CHARACTERS` | No | `18000` | Maximum context sent to the provider |

The Streamlit app loads `.env` and can read `OPENROUTER_API_KEY` from Streamlit Secrets. The ingestion and indexing commands use their own command-line options and do not require `.env`. Start the application after DVC data, chunks, and an active index are available:

```bash
uv run streamlit run src/multilingual_rag/app.py
```

The interface accepts questions in a chat box, retrieves relevant article chunks, sends the grounded context to OpenRouter, and displays the answer with source passages, page numbers, and article links. The sidebar controls the number of sources and can clear the local chat history.

### Troubleshooting

- **Missing configuration:** set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.
- **Missing source data:** run `uv run dvc pull` after configuring access to the DVC remote.
- **Missing chunks or index:** run `uv run rag-ingest` followed by `uv run rag-index`.
- **Embedding model mismatch:** rebuild the index with the same model and revision configured for the application.
- **OpenRouter failure:** verify the model identifier, API key, endpoint, account access, and timeout setting.
- **Non-fatal `torchvision` messages:** Streamlit's file watcher can inspect optional dependency modules; these messages are unrelated to the text-only E5 retrieval path. If needed, disable file watching with `--server.fileWatcherType none`.


# TODO

- There seem to some lines getting cut at the articles when extracting text from Verbraucherzentrale PDFs.
- May be should run the embedding and indexing in a docker container for faster app startup, since loading the models and embeddings can be time-consuming.