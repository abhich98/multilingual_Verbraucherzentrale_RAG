"""Export the golden dataset to the input formats of Ragas and Giskard RAGET."""

from __future__ import annotations

from typing import Any

UNANSWERABLE_CONTEXT = ""


def to_ragas(item: dict[str, Any]) -> dict[str, Any]:
    """Ragas >= 0.2 ``SingleTurnSample`` fields for a reference (golden) sample.

    At evaluation time add ``response`` (your RAG answer) and
    ``retrieved_contexts`` (the chunk texts your retriever returned).
    """
    return {
        "user_input": item["question"],
        "reference": item["reference_answer"],
        "reference_contexts": item.get("reference_contexts", []),
    }


def to_giskard(item: dict[str, Any]) -> dict[str, Any]:
    """Row of a Giskard RAGET ``QATestset`` (load with ``QATestset.load(path)``)."""
    contexts = item.get("reference_contexts", [])
    return {
        "id": item["id"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "reference_context": "\n\n".join(contexts) if contexts else UNANSWERABLE_CONTEXT,
        "conversation_history": [],
        "metadata": {
            "question_type": item["question_type"],
            "answerable": item["answerable"],
            "language": item["language"],
            "reference_chunk_ids": item.get("reference_chunk_ids", []),
            "source_titles": item.get("source_titles", []),
        },
    }
