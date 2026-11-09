"""Core data types.

Two families live here:

* ``Extracted*`` / ``Decision`` are Pydantic models. They double as the JSON
  schemas we hand to the LLM (structured outputs), so the model can only answer
  in shapes our code already understands.
* ``Memory`` / ``Source`` / ``Conflict`` are plain dataclasses for what we store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Kind(StrEnum):
    DECISION = "decision"  # "We chose Postgres over Mongo for billing."
    FACT = "fact"  # "The EU cluster runs in eu-west-1."
    PROCESS = "process"  # "Hotfixes need one approval and a passing CI run."
    OWNERSHIP = "ownership"  # "Priya owns the payments service."
    GLOSSARY = "glossary"  # "'Activation' means a user created their first agent."


class Status(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by newer knowledge; kept for history
    CONFLICTED = "conflicted"  # contradicts another memory; needs a human


# --- LLM-facing schemas -------------------------------------------------------


class ExtractedMemory(BaseModel):
    kind: Kind
    subject: str = Field(description="Short lowercase topic, e.g. 'deploy process'.")
    statement: str = Field(description="One self-contained claim, understandable on its own.")
    evidence: str = Field(description="Exact quote copied verbatim from the text.")


class ExtractionResult(BaseModel):
    memories: list[ExtractedMemory]


class Decision(BaseModel):
    action: Literal["add", "duplicate", "update", "conflict"]
    target_id: str | None = Field(
        default=None, description="Existing memory id; required unless action is 'add'."
    )
    reason: str


# --- Stored records -----------------------------------------------------------


@dataclass
class Source:
    doc_path: str
    locator: str  # e.g. "L12-L30"
    quote: str
    doc_date: str  # ISO date (YYYY-MM-DD) of the document the claim came from


@dataclass
class Memory:
    id: str
    kind: Kind
    subject: str
    statement: str
    status: Status
    created_at: str
    updated_at: str
    superseded_by: str | None = None
    sources: list[Source] = field(default_factory=list)

    @property
    def latest_date(self) -> str:
        return max((s.doc_date for s in self.sources), default="")


@dataclass
class Conflict:
    id: str
    memory_a: str
    memory_b: str
    reason: str
    resolved: bool = False
