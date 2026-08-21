"""Explicit evaluation set for the Kestrel Ridge RAG pipeline.

Every question's ground truth is drawn directly from
data/raw/kestrel_ridge_report.txt, then hand-mapped to which chunk(s) in
data/processed/chunks.json actually contain that evidence. Chunks overlap by
30 words (see Step 2), so some facts legitimately sit in more than one
chunk — expected_chunk_ids lists every chunk that would count as a correct
retrieval, not just one.

Two independent things are recorded per question, matching the two stages
being evaluated separately:
    - expected_chunk_ids: which chunk(s) retrieval SHOULD surface as
      evidence (empty for deliberately unanswerable questions — there is no
      correct evidence to retrieve).
    - expected_key_facts: substrings that MUST all appear in a correct
      generated answer (only meaningful for answerable questions).
"""

EVAL_QUESTIONS = [
    {
        "id": "mineral",
        "question": "What mineral is being mined?",
        "type": "answerable",
        "expected_chunk_ids": [
            "kestrel_ridge_report.txt::0",
            "kestrel_ridge_report.txt::1",
        ],
        "expected_key_facts": ["spodumene"],
    },
    {
        "id": "concentrate_grade",
        "question": "What concentrate grade is produced?",
        "type": "answerable",
        "expected_chunk_ids": ["kestrel_ridge_report.txt::1"],
        "expected_key_facts": ["5.8%"],
    },
    {
        "id": "resource_grade",
        "question": "What is the average grade of the resource estimate?",
        "type": "answerable",
        "expected_chunk_ids": ["kestrel_ridge_report.txt::1"],
        "expected_key_facts": ["1.24%", "0.09%"],
    },
    {
        "id": "workforce_count",
        "question": "How many full-time workers does the operation employ?",
        "type": "answerable",
        "expected_chunk_ids": ["kestrel_ridge_report.txt::2"],
        "expected_key_facts": ["340"],
    },
    {
        "id": "site_danger_paraphrase",
        # Paraphrase: no keyword overlap with "LTIFR" or "safety induction",
        # so a correct retrieval here demonstrates semantic (not lexical)
        # matching, per Step 4.
        "question": "Is the site dangerous for the people who work there?",
        "type": "paraphrase",
        "expected_chunk_ids": [
            "kestrel_ridge_report.txt::2",
            "kestrel_ridge_report.txt::3",
        ],
        "expected_key_facts": ["0.9", "1.6"],
    },
    {
        "id": "ceo_unanswerable",
        # Deliberately absent from the source document.
        "question": "Who is the CEO of Northgate Resources?",
        "type": "unanswerable",
        "expected_chunk_ids": [],
        "expected_key_facts": [],
    },
    {
        "id": "ticker_unanswerable",
        # Deliberately absent from the source document.
        "question": "What is Northgate Resources' stock ticker symbol?",
        "type": "unanswerable",
        "expected_chunk_ids": [],
        "expected_key_facts": [],
    },
]
