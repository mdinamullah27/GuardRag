from typing import Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    status: Literal["ready", "already_indexed"] = "ready"
    reused: bool = False
    # Present when extraction found problems (e.g. some pages have no usable text layer).
    warning: Optional[str] = None


class ChatRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    current_page: Optional[int] = Field(default=None, ge=1)
    selected_text: Optional[str] = Field(default=None, max_length=4000)


class Source(BaseModel):
    page: int
    chunk_id: int = 0


class ChatResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: Optional[str]
    cache_hit: bool
    evidence_sufficient: bool
    sources: list[Source] = Field(default_factory=list)
    reason: Optional[str] = None
    # Retrieval strategy that produced this answer (routing.QueryMode).
    query_mode: str = "NORMAL"
    # For exhaustive-class questions: "complete" only when the whole text was searched.
    completeness: Optional[Literal["complete", "partial"]] = None


class EvidenceVerdict(BaseModel):
    sufficient: bool
    reason: str = Field(min_length=1)


class ModeChoice(BaseModel):
    """LLM routing fallback output."""

    mode: str = "NORMAL"


class SubQueries(BaseModel):
    """LLM decomposition output for multi-hop questions."""

    queries: list[str] = Field(default_factory=list)


class LlmSettingsUpdate(BaseModel):
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = Field(default=None, min_length=1)
    llm_base_url: Optional[str] = Field(default=None, min_length=1)


class LlmSettingsView(BaseModel):
    llm_model: str
    llm_base_url: str
    llm_api_key_masked: Optional[str] = None
