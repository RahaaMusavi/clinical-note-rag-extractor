from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever import load_guidelines, GuidelineRetriever
from src.extractor import ClinicalNoteExtractor
from src.schemas import CareExtractionResult

app = FastAPI(title="Clinical Shift-Note RAG Extractor")

chunks = load_guidelines("data/guidelines.md")
retriever = GuidelineRetriever(chunks)
extractor = ClinicalNoteExtractor()


class NoteRequest(BaseModel):
    note: str


@app.post("/extract", response_model=CareExtractionResult)
def extract_care_actions(req: NoteRequest):
    top_chunks = retriever.retrieve_hybrid_reranked(req.note, top_k=3)
    return extractor.extract(req.note, top_chunks)
