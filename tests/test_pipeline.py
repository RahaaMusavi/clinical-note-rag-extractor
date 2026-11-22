from unittest.mock import MagicMock
from src.retriever import load_guidelines, normalize_clinical_jargon, tokenize, GuidelineChunk
from src.extractor import ClinicalNoteExtractor
from src.schemas import CareExtractionResult


def test_load_guidelines_parses_rule_ids():
    chunks = load_guidelines("data/guidelines.md")
    assert len(chunks) >= 10
    assert all(c.rule_id.startswith("RULE-") for c in chunks)


def test_jargon_normalization_expands_shorthand():
    note = "Checked her fingerstick before lunch, came back at 61."
    expanded = normalize_clinical_jargon(note)
    assert "blood glucose" in expanded
    assert "hypoglycemia" in expanded


def test_tokenize_removes_stopwords():
    tokens = tokenize("She tripped over her slippers in the room")
    assert "she" not in tokens
    assert "room" not in tokens
    assert "tripped" in tokens
    assert "slippers" in tokens


def test_citation_guard_forces_abstention_on_unretrieved_id():
    extractor = ClinicalNoteExtractor(api_key="dummy-key")
    fake_parsed = CareExtractionResult(
        observations=["Routine morning vitals"],
        required_actions=["Fabricated action"],
        cited_rule_ids=["RULE-NONEXISTENT-99"],
        abstained=False,
        abstention_reason=None,
    )
    mock_msg = MagicMock()
    mock_msg.parsed = fake_parsed
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    extractor.client.beta.chat.completions.parse = MagicMock(
        return_value=MagicMock(choices=[mock_choice])
    )

    retrieved = [GuidelineChunk(rule_id="RULE-FALL-01", title="Falls", text="[RULE-FALL-01] Falls: ...")]
    result = extractor.extract("Patient ate breakfast well.", retrieved)

    assert result.abstained is True
    assert result.cited_rule_ids == []
    assert result.required_actions == []
