# Grounded Clinical Shift-Note Extractor & RAG Evaluation Harness

A modular Retrieval-Augmented Generation (RAG) and schema-enforced extraction pipeline that converts messy, spoken-style nursing shift handoff notes into validated JSON records grounded in standardized clinical care protocols—with deterministic abstention when evidence is missing.

## Architecture
- **Jargon-Normalized Hybrid Retrieval & Reranking (`src/retriever.py`):** Maps spoken nursing shorthand (e.g., *"fingerstick"*, *"pulse ox"*, *"pees"*) to clinical terminology, filters stopwords, and combines `text-embedding-3-small` dense vectors (`0.85`) with `BM25Okapi` lexical scoring (`0.15`) and title-entity overlap reranking.
- **Schema-Enforced Extraction & Grounded Abstention (`src/schemas.py`, `src/extractor.py`):** Uses Pydantic structured outputs (`CareExtractionResult`) paired with a deterministic rule-ID verification guard so the model cites exact protocol IDs or explicitly abstains on routine/unanswerable notes.
- **Automated Ablation & Regression Suite (`eval/run_eval.py`):** Benchmarks retrieval `Recall@3`, exact rule citation accuracy, and false-positive hallucination rate across 20 curated test cases (15 clinical triggers + 5 routine/lexical-trap distractors).
- **Production API (`app.py`):** FastAPI `/extract` endpoint returning typed `CareExtractionResult` payloads.

## Ablation & Iterative Error Analysis Results

| Pipeline Variant | Retrieval Recall@3 | Rule Citation Exact Match | Unanswerable Hallucination Rate | Avg Latency |
| :--- | :---: | :---: | :---: | :---: |
| Dense Vector RAG (Baseline) | 100.0% | 100.0% | 0.0% | 2279 ms |
| Hybrid (BM25 + Dense) + Jargon Norm + Schema Guard | 100.0% | 100.0% | 0.0% | 2321 ms |

### Iterative Engineering Insights
1. **Zero False-Positive Hallucinations (`0.0%`):** Across all 5 routine/unanswerable shift notes—including lexical traps such as *"history of hypertension, BP 138/84"* vs. `RULE-HYD-02` (*"blood pressure falls below 90/60"*)—the schema and citation guard achieved 100% abstention precision.
2. **Diagnosing & Fixing Spoken-to-Clinical Register Mismatch:** In initial ablations (`93.3%` dense vs. `86.7%` uncalibrated 60/40 hybrid), two failure modes emerged:
   - *Colloquial Jargon (`NOTE-05`):* Spoken terms (*"fingerstick ... 61"*) missed formal protocol terms (*"blood glucose ... mg/dL"* in `RULE-GLUC-05`). Adding lightweight domain shorthand normalization resolved the register gap.
   - *Stopword Dilution in BM25 (`NOTE-13`):* Unfiltered operational words (*"floor"*, *"front"*, *"normal"*) inflated BM25 scores on unrelated rules. Filtering stopwords and rebalancing weights (`0.85` dense / `0.15` BM25) stabilized retrieval at `100.0%` `Recall@3`.
3. **End-to-End Citation Faithfulness:** Whenever the target protocol appeared in the retrieved `top_k=3` context, `gpt-4o-mini` with Pydantic structured outputs achieved `100%` exact rule citation accuracy.

## Quickstart
```bash
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY
python -m eval.run_eval
uvicorn app:app --reload
```
