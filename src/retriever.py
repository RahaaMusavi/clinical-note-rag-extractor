import os
import re
from dataclasses import dataclass
from typing import List
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "was", "were", "be", "been", "she", "he", "her", "his", "mrs",
    "mr", "ms", "room", "just", "right", "now", "about", "before", "after", "no",
    "not", "any", "into", "over", "from", "that", "this", "it", "they", "them"
}

CLINICAL_EXPANSIONS = {
    r"\bfingerstick\b": "fingerstick blood glucose mg/dL hypoglycemia",
    r"\bpulse ox\b": "pulse oximetry oxygen saturation SpO2",
    r"\bpees?\b|\bpeeing\b": "urination dysuria urinary",
    r"\btripped\b|\bslipped\b": "tripped slipped fall",
    r"\bblood thinners?\b": "blood thinners anticoagulants",
}


@dataclass
class GuidelineChunk:
    rule_id: str
    title: str
    text: str


def normalize_clinical_jargon(text: str) -> str:
    expanded = text
    for pattern, replacement in CLINICAL_EXPANSIONS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def load_guidelines(path: str = "data/guidelines.md") -> List[GuidelineChunk]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = [s.strip() for s in content.split("### ") if s.strip()]
    chunks = []
    for sec in sections:
        match = re.match(r"\[(RULE-[A-Z]+-\d+)\]\s+(.+?)\n(.+)", sec, re.DOTALL)
        if match:
            chunks.append(
                GuidelineChunk(
                    rule_id=match.group(1),
                    title=match.group(2).strip(),
                    text=f"[{match.group(1)}] {match.group(2).strip()}: {match.group(3).strip()}",
                )
            )
    return chunks


class GuidelineRetriever:
    def __init__(
        self,
        chunks: List[GuidelineChunk],
        embedding_model: str = "text-embedding-3-small",
    ):
        self.chunks = chunks
        self.corpus_texts = [c.text for c in chunks]
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        tokenized_corpus = [tokenize(doc) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=self.corpus_texts,
        )
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        self.doc_embeddings = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def retrieve_dense_only(self, query: str, top_k: int = 3) -> List[GuidelineChunk]:
        q_emb = self._embed_query(query)
        scores = np.dot(self.doc_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def retrieve_hybrid_reranked(
        self, query: str, candidate_k: int = 6, top_k: int = 3
    ) -> List[GuidelineChunk]:
        normalized_query = normalize_clinical_jargon(query)
        q_emb = self._embed_query(normalized_query)
        dense_scores = np.dot(self.doc_embeddings, q_emb)

        tokenized_query = tokenize(normalized_query)
        bm25_raw = np.array(self.bm25.get_scores(tokenized_query))
        bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-8)

        hybrid_scores = 0.85 * dense_scores + 0.15 * bm25_norm
        candidate_indices = np.argsort(hybrid_scores)[::-1][:candidate_k]

        query_terms = set(tokenized_query)
        reranked = []
        for idx in candidate_indices:
            title_terms = set(tokenize(self.chunks[idx].title))
            overlap_bonus = 0.04 * len(query_terms & title_terms)
            reranked.append((hybrid_scores[idx] + overlap_bonus, self.chunks[idx]))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in reranked[:top_k]]
