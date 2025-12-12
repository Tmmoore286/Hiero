from __future__ import annotations

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    query_id: str
    question: str
    ground_truth_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EvalDataset(BaseModel):
    name: str
    queries: list[EvalQuery]
    description: str | None = None
    version: str = "1.0"

