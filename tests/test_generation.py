"""Boundary checks for Step 5's prompt construction (src/generation/generate.py).

build_prompt() is pure string assembly, so these run without loading any
model. Actual generation quality (does the LLM answer correctly, does it
decline when it should) is exercised end-to-end in test_pipeline.py via the
real Step 6 evaluation, rather than duplicated here.
"""

from src.generation.generate import PROMPT_INSTRUCTIONS, build_prompt


def test_build_prompt_includes_question_and_every_chunk_id():
    chunks = [
        {"chunk_id": "doc::0", "text": "alpha fact"},
        {"chunk_id": "doc::1", "text": "beta fact"},
    ]
    prompt = build_prompt("What is alpha?", chunks)
    assert PROMPT_INSTRUCTIONS in prompt
    assert "[doc::0] alpha fact" in prompt
    assert "[doc::1] beta fact" in prompt
    assert prompt.endswith("QUESTION:\nWhat is alpha?")


def test_build_prompt_handles_no_retrieved_chunks():
    # Boundary case: retrieval found nothing to ground the answer in.
    prompt = build_prompt("Any question?", [])
    assert prompt.endswith("QUESTION:\nAny question?")
    assert "[" not in prompt  # no chunk citations when there's no evidence
