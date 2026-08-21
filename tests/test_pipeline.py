"""End-to-end boundary checks (Steps 5-7), exercised through the pipeline's
real public entry points rather than re-implemented here:

    - src.evaluation.evaluate.run_evaluation() / compute_scores() (Step 6),
      which itself calls src.generation.generate.answer_question() (Step 5)
      for every question in the eval set.
    - src.interface.ask.ask() (Step 7).

The evaluation_results fixture monkeypatches evaluate's load_* functions to
hand back the already-loaded session fixtures instead of loading the
embedding model and Qwen2.5-1.5B-Instruct a second time -- pure test-speed
plumbing, not a change to evaluate.py's own logic.
"""

import pytest

from src.evaluation import evaluate
from src.interface import ask as ask_module


@pytest.fixture(scope="module")
def evaluation_results(embeddings_data, retrieval_model, llm):
    llm_tokenizer, llm_model = llm
    patcher = pytest.MonkeyPatch()
    patcher.setattr(evaluate, "load_embeddings", lambda: embeddings_data)
    patcher.setattr(evaluate, "load_model", lambda data: retrieval_model)
    patcher.setattr(evaluate, "load_llm", lambda: (llm_tokenizer, llm_model))
    try:
        return evaluate.run_evaluation()
    finally:
        patcher.undo()


def _result(results, question_id):
    return next(r for r in results if r["id"] == question_id)


def test_answerable_question_grounded_correctly(evaluation_results):
    result = _result(evaluation_results, "concentrate_grade")
    assert result["retrieval_pass"] is True
    assert result["generation_pass"] is True


def test_paraphrased_question_retrieved_and_answered_correctly(evaluation_results):
    result = _result(evaluation_results, "site_danger_paraphrase")
    assert result["retrieval_pass"] is True
    assert result["generation_pass"] is True


def test_unanswerable_questions_are_declined_not_hallucinated(evaluation_results):
    for question_id in ("ceo_unanswerable", "ticker_unanswerable"):
        result = _result(evaluation_results, question_id)
        assert result["retrieval_pass"] is None  # no expected evidence to score
        assert result["generation_pass"] is True  # declined rather than guessed


def test_evaluation_scores_are_reproducible(evaluation_results):
    scores = evaluate.compute_scores(evaluation_results)
    assert scores["retrieval_recall_at_k"] == 1.0
    assert scores["retrieval_evaluated_count"] == 5
    assert scores["total_questions"] == 7
    assert scores["generation_accuracy"] == pytest.approx(6 / 7)


def test_ask_interface_returns_inspectable_grounded_answer(
    embeddings_data, retrieval_model, llm, monkeypatch
):
    llm_tokenizer, llm_model = llm
    # Reuse the already-loaded session models instead of letting ask()
    # trigger its own (identical) load on first call.
    monkeypatch.setitem(ask_module._state, "embeddings_data", embeddings_data)
    monkeypatch.setitem(ask_module._state, "retrieval_model", retrieval_model)
    monkeypatch.setitem(ask_module._state, "llm_tokenizer", llm_tokenizer)
    monkeypatch.setitem(ask_module._state, "llm_model", llm_model)

    result = ask_module.ask("What mineral is being mined?")

    assert result["question"] == "What mineral is being mined?"
    assert "spodumene" in result["answer"].lower()
    assert result["source_chunk_ids"]  # answer is traceable to specific chunks
