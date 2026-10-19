# Golden dataset and retrieval evaluation

`golden_dataset.jsonl` holds 40 hand-written evaluation items built from `artifacts/chunks.jsonl` (ingestion run of 2026-10-18, 837 chunks, 66 articles):

| | English | German | Total |
| --- | --- | --- | --- |
| Answerable | 20 | 10 | 30 |
| Unanswerable | 6 | 4 | 10 |

The corpus is entirely German, so the 26 English questions test cross-lingual retrieval with `multilingual-e5-small`. Reference answers are written in the question's language.

## Schema

| Field | Meaning |
| --- | --- |
| `id` | `q001` to `q040`; answerable items come first |
| `question` | The user query |
| `reference_answer` | Ground-truth answer, grounded strictly in the evidence |
| `reference_contexts` | Source chunk texts, one chunk per evidence quote (Ragas `reference_contexts`) |
| `reference_chunk_ids` | `chunk_id`s of those chunks |
| `source_titles` | Article titles the evidence comes from |
| `question_type` | See below |
| `answerable` | `false` for the 10 unanswerable items |
| `language` | `en` or `de` |
| `evidence` | List of `{source_path, quote, chunk_ids}`: verbatim quotes of each fact the answer needs, and **every** chunk that contains the quote |
| `related` | (unanswerable only) bait chunks that look relevant but don't answer the question; not used for metrics |
| `notes` | What an item is designed to test |

### Question types

Answerable: `simple` (10), `numeric` (6), `multi_hop` (5, several facts from one article), `cross_document` (4, facts from two articles), `conditional` (3, the answer depends on a condition in the text), `distracting` (2, the question invites a plausible wrong answer).

Unanswerable:
- `out_of_corpus` (1): the topic isn't covered.
- `distractor` (4): words or numbers from the question appear in the corpus but don't answer it, e.g. the Deutschlandticket price, or the 5-year update duty mistaken for a warranty.
- `partial_coverage` (3): the article covers the topic but not the asked detail.
- `false_premise` (2): the corpus contradicts the question. These two carry `evidence`: the expected answer is a correction, and retrieval is scored on finding the correcting chunk.

Two of the `partial_coverage` items exist because of extraction gaps: the medium water hardness range was lost between chunks, and the collapsed accordions in the advertising-mail article are missing. If you fix ingestion and the text shows up, convert those items to answerable.

## Retrieval metrics

```bash
uv run rag-eval retrieval                 # K = 1 3 5 10 by default
uv run rag-eval retrieval --k 1 3 5 10 20
```

Metrics are computed on the 32 items with evidence (30 answerable plus 2 false-premise). Each evidence quote is a required fact, and retrieving **any** chunk that contains it satisfies that fact. This matters because chunks overlap by 150 characters, so a quote can sit in two chunks.

- **Recall@K**: share of an item's required facts found in the top K, averaged over items.
- **Hit@K**: share of items with at least one relevant chunk in the top K.
- **MRR**: mean of 1 / rank of the first relevant chunk (0 if none is found within max K).
- **DocRecall@K**: share of gold articles with any chunk in the top K.

Results are broken down by question type and language. They are written to `eval/results/retrieval_<timestamp>.json`, plus a per-query file with ranks and retrieved chunks for error analysis.

### Why quotes and not only chunk IDs

`chunk_id` hashes the source path, page, chunk index and text, so it changes whenever chunk size, overlap or text cleaning changes. At evaluation time the script re-matches every quote against the chunks of the **active index**. Re-chunking therefore doesn't silently invalidate the gold labels. If a quote can no longer be found, the run stops and names the item.

After re-ingesting, refresh the stored IDs and contexts, then rebuild the index:

```bash
uv run rag-ingest
uv run rag-eval refresh --chunks artifacts/chunks.jsonl
uv run rag-index
```

## Ragas and Giskard

```bash
uv run rag-eval export        # writes eval/exports/
```

- `ragas_golden.jsonl`: answerable items with Ragas ≥ 0.2 field names (`user_input`, `reference`, `reference_contexts`). At eval time, add `response` and `retrieved_contexts` from your pipeline.
- `ragas_golden_unanswerable.jsonl`: the 10 unanswerable items. Score these separately (refusal or correction check). Faithfulness and context metrics are not meaningful with empty reference contexts.
- `giskard_qatestset.jsonl`: Giskard RAGET `QATestset` rows (`question`, `reference_answer`, `reference_context`, `conversation_history`, `metadata`). Load it with `QATestset.load(...)`.
