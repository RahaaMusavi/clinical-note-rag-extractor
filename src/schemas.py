from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CareExtractionResult(BaseModel):
    resident_identifier: Optional[str] = Field(
        default=None,
        description="Resident name and/or room number mentioned in the note."
    )
    event_category: Optional[
        Literal[
            "fall",
            "hydration",
            "skin_integrity",
            "medication",
            "hypoglycemia",
            "respiratory",
            "infection_uti",
            "swallowing",
            "pain",
            "fever_sepsis",
            "behavioral",
            "gastrointestinal",
        ]
    ] = Field(
        default=None,
        description="Category matching the retrieved guideline, or null if no guideline applies."
    )
    clinical_observations: List[str] = Field(
        default_factory=list,
        description="Factual symptoms, vitals, or events explicitly stated in the note."
    )
    required_actions: List[str] = Field(
        default_factory=list,
        description="Actionable steps strictly grounded in the retrieved guideline rules."
    )
    urgency_level: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Urgency derived from the matching guideline criteria, or null if abstaining."
    )
    cited_rule_ids: List[str] = Field(
        default_factory=list,
        description="Exact rule IDs (e.g., 'RULE-FALL-01') supporting the extracted actions."
    )
    abstained: bool = Field(
        default=False,
        description="True if none of the retrieved guidelines apply to the note."
    )
    abstention_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation when abstained is True."
    )
