from __future__ import annotations
import os
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from src.schemas import CareExtractionResult
from src.retriever import GuidelineChunk

load_dotenv(override=True)

SYSTEM_PROMPT = """You are a clinical documentation compliance extractor.
Given a short nursing shift note and a set of retrieved care protocol guidelines:
1. Extract factual observations mentioned in the note.
2. Check if ANY retrieved guideline directly governs the clinical situation described.
3. If a guideline applies, extract the required actions strictly from that guideline and cite its exact rule ID (e.g., RULE-FALL-01).
4. CRITICAL ABSTENTION RULE: If the note describes routine daily activities, non-clinical requests, or normal vital signs that do NOT trigger any retrieved guideline protocol, set `abstained = true`, leave `required_actions` and `cited_rule_ids` empty, and explain why in `abstention_reason`. Never fabricate rules or force a match."""


class ClinicalNoteExtractor:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and not base_url:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=resolved_key or "ollama",
        )

    def extract(self, note: str, retrieved_chunks: List[GuidelineChunk]) -> CareExtractionResult:
        context_block = "\n\n".join([c.text for c in retrieved_chunks])
        user_prompt = f"""### Retrieved Care Guidelines:
{context_block}

### Nursing Shift Note:
\"\"\"{note}\"\"\"
"""
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=CareExtractionResult,
        )
        result = completion.choices[0].message.parsed

        valid_ids = {c.rule_id for c in retrieved_chunks}
        result.cited_rule_ids = [rid for rid in result.cited_rule_ids if rid in valid_ids]
        if not result.cited_rule_ids and not result.abstained:
            result.abstained = True
            result.required_actions = []
            result.abstention_reason = "No retrieved guideline ID could be verified."

        return result
