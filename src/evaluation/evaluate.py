"""Step 6: evaluate retrieval and generation quality separately.

RAG can fail two independent ways — the wrong evidence gets retrieved, or
the right evidence gets retrieved but the LLM uses it badly (or ignores it)
— and those two failure modes need two different fixes (better retrieval
vs. a better prompt/model). Scoring only the final answer conflates them.
This module scores each stage on its own for every question in the eval set.

Reuses Steps 4 and 5 directly, no logic duplicated:
    - src.retrieval.search: load_embeddings, load_model (Step 4)
    - src.generation.generate: load_llm, answer_question (Step 5), which
      itself calls Step 4's search() internally.

answer_question() is called once per question, and its two outputs are
graded independently:
    - retrieved_evidence -> evaluate_retrieval() -> Recall@k style pass/fail
    - answer              -> evaluate_generation() -> factual pass/fail
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.dataset import EVAL_QUESTIONS  # noqa: E402
from src.generation.generate import DEFAULT_TOP_K, answer_question, load_llm  # noqa: E402
from src.retrieval.search import load_embeddings, load_model  # noqa: E402

REPORT_FILE = BASE_DIR / "reports" / "evaluation_report.md"
RESULTS_FILE = BASE_DIR / "reports" / "evaluation_results.json"

# Deterministic, keyword-based check for whether the model declined to
# answer rather than hallucinating an answer for an unanswerable question.
# No second LLM is used as a judge — the goal is a transparent, reproducible
# signal that can be inspected and re-run identically, not a graded opinion.
DECLINE_PHRASES = [
    "don't know",
    "do not know",
    "not mentioned",
    "not provided",
    "not possible to determine",
    "cannot determine",
    "can't determine",
    "no information",
    "not specified",
    "does not contain",
    "doesn't contain",
    "unable to determine",
    "not available in",
    "not stated",
]


def evaluate_retrieval(retrieved_evidence: list[dict], expected_chunk_ids: list[str]) -> bool | None:
    """Recall@k for one question: did any expected chunk land in the top-k?

    Returns None when the question has no expected evidence at all (a
    deliberately unanswerable question) — "correct retrieval" isn't
    meaningful there, so it's excluded from the retrieval score rather than
    counted as a failure.
    """
    if not expected_chunk_ids:
        return None
    retrieved_ids = {chunk["chunk_id"] for chunk in retrieved_evidence}
    return bool(retrieved_ids & set(expected_chunk_ids))


def evaluate_generation(answer: str, question_type: str, expected_key_facts: list[str]) -> bool:
    """Check the generated answer against ground truth, deterministically.

    Answerable/paraphrase questions pass only if EVERY expected key fact
    appears in the answer (substring match, case-insensitive). Unanswerable
    questions pass only if the answer contains a decline phrase instead of
    inventing specifics.
    """
    answer_lower = answer.lower()
    if question_type == "unanswerable":
        return any(phrase in answer_lower for phrase in DECLINE_PHRASES)
    return all(fact.lower() in answer_lower for fact in expected_key_facts)


def run_evaluation(top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Run every eval question through the Step 5 pipeline and grade both stages."""
    embeddings_data = load_embeddings()
    retrieval_model = load_model(embeddings_data)
    llm_tokenizer, llm_model = load_llm()

    results = []
    for item in EVAL_QUESTIONS:
        outcome = answer_question(
            item["question"],
            embeddings_data,
            retrieval_model,
            llm_tokenizer,
            llm_model,
            top_k=top_k,
        )
        retrieval_pass = evaluate_retrieval(outcome["retrieved_evidence"], item["expected_chunk_ids"])
        generation_pass = evaluate_generation(outcome["answer"], item["type"], item["expected_key_facts"])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "type": item["type"],
                "expected_chunk_ids": item["expected_chunk_ids"],
                "expected_key_facts": item["expected_key_facts"],
                "retrieved_chunk_ids": [c["chunk_id"] for c in outcome["retrieved_evidence"]],
                "retrieved_scores": {
                    c["chunk_id"]: c["similarity_score"] for c in outcome["retrieved_evidence"]
                },
                "answer": outcome["answer"],
                "retrieval_pass": retrieval_pass,
                "generation_pass": generation_pass,
            }
        )
    return results


def compute_scores(results: list[dict]) -> dict:
    """Aggregate Recall@k for retrieval and accuracy for generation, separately."""
    scored_retrieval = [r for r in results if r["retrieval_pass"] is not None]
    retrieval_hits = sum(1 for r in scored_retrieval if r["retrieval_pass"])
    generation_hits = sum(1 for r in results if r["generation_pass"])
    return {
        "retrieval_recall_at_k": (retrieval_hits / len(scored_retrieval)) if scored_retrieval else None,
        "retrieval_evaluated_count": len(scored_retrieval),
        "retrieval_hits": retrieval_hits,
        "generation_accuracy": generation_hits / len(results),
        "generation_hits": generation_hits,
        "total_questions": len(results),
    }


def _fmt_pass(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def write_report(results: list[dict], scores: dict, top_k: int, report_file: Path = REPORT_FILE) -> None:
    """Write a human-readable Markdown report showing every question's evidence,
    answer, and pass/fail for both stages, plus the aggregate scores."""
    lines = [
        "# RAG Pipeline Evaluation Report (Step 6/8)",
        "",
        f"Source document: `kestrel_ridge_report.txt` | top_k = {top_k} | "
        f"generation model = Qwen2.5-1.5B-Instruct",
        "",
        "Retrieval and generation are graded independently for every "
        "question: a question can have correct retrieval with a wrong "
        "answer, or vice versa, and this report keeps the two visible "
        "separately rather than collapsing them into one score.",
        "",
        "## Aggregate scores",
        "",
        f"- **Retrieval Recall@{top_k}:** {scores['retrieval_hits']}/"
        f"{scores['retrieval_evaluated_count']} "
        f"({scores['retrieval_recall_at_k']:.0%}) "
        "— questions with no expected evidence (unanswerable) are excluded",
        f"- **Generation accuracy:** {scores['generation_hits']}/"
        f"{scores['total_questions']} ({scores['generation_accuracy']:.0%})",
        "",
        "## Per-question results",
        "",
    ]

    for r in results:
        lines.append(f"### `{r['id']}` — {r['type']}")
        lines.append("")
        lines.append(f"**Question:** {r['question']}")
        lines.append("")
        expected_evidence = ", ".join(f"`{c}`" for c in r["expected_chunk_ids"]) or "_(none — unanswerable)_"
        lines.append(f"**Expected evidence:** {expected_evidence}")
        expected_facts = ", ".join(r["expected_key_facts"]) or "_(none — should decline)_"
        lines.append(f"**Expected key facts:** {expected_facts}")
        lines.append("")
        lines.append("**Retrieved chunks (top-k, ranked):**")
        for chunk_id in r["retrieved_chunk_ids"]:
            score = r["retrieved_scores"][chunk_id]
            lines.append(f"- `{chunk_id}` (similarity {score:.4f})")
        lines.append("")
        lines.append(f"**Generated answer:** {r['answer']}")
        lines.append("")
        lines.append(
            f"**Retrieval:** {_fmt_pass(r['retrieval_pass'])} &nbsp;&nbsp; "
            f"**Generation:** {_fmt_pass(r['generation_pass'])}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: list[dict], scores: dict, top_k: int) -> None:
    print("=" * 80)
    print(f"RAG EVALUATION (top_k={top_k})")
    print("=" * 80)
    for r in results:
        print(
            f"[{r['id']:<24}] retrieval={_fmt_pass(r['retrieval_pass']):<4} "
            f"generation={_fmt_pass(r['generation_pass']):<4} :: {r['question']}"
        )
    print("-" * 80)
    print(
        f"Retrieval Recall@{top_k}: {scores['retrieval_hits']}/"
        f"{scores['retrieval_evaluated_count']} ({scores['retrieval_recall_at_k']:.0%})"
    )
    print(
        f"Generation accuracy:  {scores['generation_hits']}/"
        f"{scores['total_questions']} ({scores['generation_accuracy']:.0%})"
    )


def run(top_k: int = DEFAULT_TOP_K) -> None:
    results = run_evaluation(top_k=top_k)
    scores = compute_scores(results)
    print_summary(results, scores, top_k)
    write_report(results, scores, top_k)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps({"top_k": top_k, "scores": scores, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport written to {REPORT_FILE.relative_to(BASE_DIR)}")
    print(f"Raw results written to {RESULTS_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    run()
