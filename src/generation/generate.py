"""Grounded RAG answer generation: retrieval output + a local open-source LLM.

Runtime flow:
    question -> embed question -> cosine similarity -> retrieve top-k chunks
             -> build_prompt(question, chunks) -> local instruction-tuned LLM
             -> answer

Retrieval logic is NOT duplicated here — it's imported directly from
src.retrieval.search (Step 4). This module only adds prompt construction and
local LLM inference on top of it.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.search import DEFAULT_TOP_K, load_embeddings, load_model, search  # noqa: E402

# Qwen2.5-1.5B-Instruct: ~1.5B parameters, Apache-2.0, ungated on Hugging Face,
# ~3GB of weights, strong instruction-following for its size, and runs
# comfortably on an Apple Silicon Mac (CPU or MPS) without a rented GPU.
# See README for full rationale.
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_NEW_TOKENS = 200

PROMPT_INSTRUCTIONS = (
    "Answer the QUESTION using ONLY the information in CONTEXT below.\n"
    "If CONTEXT does not contain the answer, say clearly that you don't know "
    "based on the available information — do not guess or invent facts.\n"
    "Be concise."
)


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Construct the exact text that gets sent to the LLM.

    This is the one place prompt text is assembled, so the full prompt can
    always be inspected by calling this function directly.
    """
    context = "\n\n".join(
        f"[{chunk['chunk_id']}] {chunk['text']}" for chunk in retrieved_chunks
    )
    return (
        f"{PROMPT_INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )


def load_llm(model_name: str = GENERATION_MODEL_NAME):
    """Load the local instruction-tuned model and tokenizer for inference only."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()  # inference mode: no gradients, no weight updates
    return tokenizer, model


def generate_answer(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run the prompt through the local LLM and return the generated text only."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    with torch.no_grad():  # inference only, no gradient tracking, no training
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for deterministic, reproducible answers
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def answer_question(
    question: str,
    embeddings_data: dict,
    retrieval_model,
    llm_tokenizer,
    llm_model,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Full grounded-answer pipeline: retrieve evidence, build prompt, generate answer."""
    retrieved_evidence = search(question, retrieval_model, embeddings_data, top_k=top_k)
    prompt = build_prompt(question, retrieved_evidence)
    answer = generate_answer(prompt, llm_tokenizer, llm_model)
    return {
        "question": question,
        "retrieved_evidence": retrieved_evidence,
        "prompt": prompt,
        "answer": answer,
    }


# Demo questions: some answerable from the report, one paraphrased, and one
# whose answer is deliberately absent from the source document.
DEMO_QUESTIONS = [
    "What mineral is being mined?",
    "What concentrate grade is produced?",
    "Is the site dangerous for the people who work there?",  # paraphrase
    "Who is the CEO of Northgate Resources?",  # not present in the report
]


def print_result(result: dict) -> None:
    print("=" * 80)
    print(f"QUESTION\n{result['question']}\n")

    print("RETRIEVED EVIDENCE")
    for e in result["retrieved_evidence"]:
        print(
            f"  chunk_id={e['chunk_id']} similarity_score={e['similarity_score']:.4f}"
        )
    print()

    print("PROMPT")
    print(result["prompt"])
    print()

    print("MODEL ANSWER")
    print(result["answer"])
    print()


def run_demo(top_k: int = DEFAULT_TOP_K) -> None:
    embeddings_data = load_embeddings()
    retrieval_model = load_model(embeddings_data)
    llm_tokenizer, llm_model = load_llm()

    for question in DEMO_QUESTIONS:
        result = answer_question(
            question, embeddings_data, retrieval_model, llm_tokenizer, llm_model, top_k=top_k
        )
        print_result(result)


if __name__ == "__main__":
    run_demo()
