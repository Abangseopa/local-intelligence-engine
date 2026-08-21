"""Step 7: a single ask() entry point that orchestrates Steps 1-6 into one
usable pipeline.

No retrieval, prompt-building, or generation logic lives here — this module
only wires together what already exists:

    question -> embed_query() + cosine similarity (Step 4, via Step 5's
    answer_question()) -> retrieved top-k chunks -> build_prompt() (Step 5)
    -> local Qwen2.5-1.5B-Instruct (Step 5) -> grounded answer

Programmatic use:
    from src.interface.ask import ask
    result = ask("What concentrate grade is produced?")

Command-line use:
    python3 src/interface/ask.py "What concentrate grade is produced?"
    python3 src/interface/ask.py            # interactive prompt loop
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.generation.generate import DEFAULT_TOP_K, answer_question, load_llm  # noqa: E402
from src.retrieval.search import load_embeddings, load_model  # noqa: E402

# Loaded once per process and reused across calls to ask(). Loading the
# embedding model and Qwen2.5-1.5B-Instruct takes several seconds, so
# reloading them on every question would make interactive use unusably
# slow. This is resource management for the interface layer, not new
# pipeline logic — the models themselves are still loaded via Steps 4/5's
# own load_model()/load_llm() functions.
_state: dict = {}


def _get_state() -> dict:
    if not _state:
        embeddings_data = load_embeddings()
        _state["embeddings_data"] = embeddings_data
        _state["retrieval_model"] = load_model(embeddings_data)
        _state["llm_tokenizer"], _state["llm_model"] = load_llm()
    return _state


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Ask one question and get a grounded answer, end to end.

    Returns a dict with the question, the final answer, and the source
    chunk IDs used as evidence, so the answer stays inspectable rather than
    being a black box.
    """
    state = _get_state()
    result = answer_question(
        question,
        state["embeddings_data"],
        state["retrieval_model"],
        state["llm_tokenizer"],
        state["llm_model"],
        top_k=top_k,
    )
    return {
        "question": result["question"],
        "answer": result["answer"],
        "source_chunk_ids": [chunk["chunk_id"] for chunk in result["retrieved_evidence"]],
    }


def print_answer(result: dict) -> None:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources: {', '.join(result['source_chunk_ids'])}")
    print()


# One of each kind, for a quick sanity check of the whole pipeline through
# this interface: answerable, paraphrased, and deliberately unanswerable.
DEMO_QUESTIONS = [
    "What concentrate grade is produced?",
    "Is the site dangerous for the people who work there?",  # paraphrase
    "Who is the CEO of Northgate Resources?",  # not present in the report
]


def run_demo() -> None:
    for question in DEMO_QUESTIONS:
        print_answer(ask(question))


def main() -> None:
    if len(sys.argv) > 1:
        print_answer(ask(" ".join(sys.argv[1:])))
        return

    print("Local Intelligence Engine — ask a question about the loaded document(s).")
    print("Type a question and press Enter. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        print_answer(ask(question))


if __name__ == "__main__":
    main()
