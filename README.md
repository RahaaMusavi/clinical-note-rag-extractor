# Clinical Shift-Note Extractor & Grounded RAG Benchmark

Extracts structured nursing care actions from spoken-style shift handoff notes and links each action to an explicit rule in `data/guidelines.md`. If no retrieved guideline applies to the note, the extractor abstains rather than guessing.

## Problem & Design Choices

Spoken nursing shift notes and written clinical protocols operate in different linguistic registers. A shift note says *"checked her fingerstick and it came back at 61"* or *"tripped over her slippers"*, whereas a protocol specifies *"blood glucose below 70 mg/dL"* or *"unwitnessed fall assessment"*.

To handle this without fabricating citations, the pipeline separates retrieval, structured generation, and post-hoc verification:

- **Retrieval (`src/retriever.py`):** Expands common nursing shorthand via deterministic regex rules, strips operational stopwords, and ranks guidelines using weighted dense + sparse fusion (`0.85 * cosine + 0.15 * bm25`) followed by a title-token overlap adjustment.
- **Structured Extraction (`src/schemas.py`, `src/extractor.py`):** Uses `gpt-4o-mini` with Pydantic structured outputs (`CareExtractionResult`) at `temperature=0.0`.
- **Deterministic Citation Guard (`src/extractor.py`):** Intersects the model's `cited_rule_ids` against the rule IDs actually returned by the retriever. Any unverified ID is stripped; if none remain, `abstained` is forced to `True`.
- **API (`app.py`):** Exposes `POST /extract` via FastAPI.

## Evaluation & Error Analysis

Benchmarked via `python -m eval.run_eval` on 20 synthetic shift notes (`data/eval_set.json`): 15 notes requiring specific care protocols and 5 routine notes designed as lexical traps (e.g., mentioning *"history of hypertension, BP 138/84"* when the hypotension rule triggers at *`< 90/60`*).

| Pipeline Variant | Retrieval Recall@3 | Rule Citation Exact Match | Unanswerable Hallucination Rate | Avg Latency |
| :--- | :---: | :---: | :---: | :---: |
| Dense Vector Baseline (`text-embedding-3-small`) | 100.0% | 100.0% | 0.0% | 2279 ms |
| Hybrid (BM25 + Dense) + Jargon Norm + Guard | 100.0% | 100.0% | 0.0% | 2321 ms |

### What Failed During Iteration
In the first ablation pass, a naive 60/40 dense-BM25 split scored **86.7%** Recall@3 (below the **93.3%** dense baseline) due to two specific errors:
1. **Register gap (`NOTE-05`):** *"fingerstick ... 61"* shared almost no surface vocabulary with `RULE-GLUC-05` (*"blood glucose ... mg/dL"*). Adding domain shorthand expansion before embedding and tokenization resolved the miss.
2. **Stopword dilution (`NOTE-13`):** Common shift words (*"floor"*, *"front"*, *"normal"*) gave unrelated rules high BM25 scores and pushed `RULE-FALL-01` from rank #2 to #4. Filtering non-clinical stopwords and shifting fusion weights to 85/15 fixed the ranking instability.

## Project Layout

```text
├── app.py                  # FastAPI /extract endpoint
├── data/
│   ├── guidelines.md       # Standardized clinical care protocols
│   └── eval_set.json       # 15 clinical triggers + 5 abstention distractors
├── eval/
│   └── run_eval.py         # Ablation & citation accuracy benchmark
├── src/
│   ├── retriever.py        # Jargon normalization, BM25 + dense fusion, reranker
│   ├── extractor.py        # Pydantic structured extraction + citation guard
│   └── schemas.py          # CareExtractionResult schema
└── tests/
    └── test_pipeline.py    # Offline unit tests for parser, retriever utils, and guard
```

## Setup & Usage

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
pytest                 # run offline unit tests
python -m eval.run_eval
uvicorn app:app --reload
```
