"""The two LLM jobs orgmem needs, behind small interfaces.

1. **Extract**: read a chunk of text and pull out durable organisational knowledge.
2. **Judge**: compare a new memory with similar existing ones and decide whether
   it is new, a duplicate, an update, or a contradiction.

The pipeline depends only on the ``Extractor`` and ``Judge`` protocols, so tests
use deterministic fakes and you can swap in another model provider without
touching the rest of the code.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .models import Decision, ExtractedMemory, ExtractionResult, Memory

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"  # extraction is routine reading; raise via ORGMEM_EFFORT if needed

# Models that accept server-side refusal fallbacks (`fallbacks="default"`) on the Claude API.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class Extractor(Protocol):
    def extract(self, text: str, doc_path: str, doc_date: str) -> list[ExtractedMemory]: ...


class Judge(Protocol):
    def decide(self, new: ExtractedMemory, new_date: str, candidates: list[Memory]) -> Decision: ...


class LLMError(RuntimeError):
    pass


EXTRACT_SYSTEM = """\
You turn a company's internal documents into shared organisational memory: \
short, citable facts that any teammate or AI assistant in the company can rely on.

Extract only durable knowledge someone would still need weeks from now:
- decision: a choice the team made, ideally with its reason
- fact: how a system, product, customer or the business actually is
- process: how work gets done (steps, rules, approvals)
- ownership: who owns or is the contact for something
- glossary: what an internal term or acronym means

Skip small talk, one-off status updates, open questions, and opinions nobody adopted.

For each memory:
- statement: one claim, self-contained. Replace pronouns and vague references with \
names so it reads correctly without the original document.
- subject: a short lowercase topic shared by related claims (e.g. "deploy process", \
"billing database"), so different documents about the same thing line up.
- evidence: a short quote copied exactly, character for character, from the text, that \
supports the statement. Claims you cannot support with a verbatim quote are left out.

Return an empty list if the text holds nothing durable."""

JUDGE_SYSTEM = """\
You maintain a company's shared memory. A new claim has arrived; compare it to the \
existing memories listed and choose one action:

- duplicate: an existing memory already says the same thing (wording may differ).
- update: the new claim replaces an existing one because things changed (a later \
decision, a new owner, a revised process) and the new claim is dated the same day or later.
- conflict: the new claim contradicts an existing one and the dates do not show which \
is current (e.g. the new claim is older, or the two documents disagree on the same day). \
A human must decide.
- add: none of the above; the claim is new knowledge.

Set target_id to the id of the existing memory for duplicate, update or conflict, and \
leave it null for add. Give a one-sentence reason."""


class AnthropicLLM:
    """Implements both ``Extractor`` and ``Judge`` with Claude structured outputs."""

    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        import anthropic  # imported lazily so read-only commands work without the SDK set up

        self.model = model or os.environ.get("ORGMEM_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("ORGMEM_EFFORT", DEFAULT_EFFORT)
        self.client = client or anthropic.Anthropic()

    def extract(self, text: str, doc_path: str, doc_date: str) -> list[ExtractedMemory]:
        prompt = f'<document path="{doc_path}" date="{doc_date}">\n{text}\n</document>'
        return self._parse(EXTRACT_SYSTEM, prompt, ExtractionResult).memories

    def decide(self, new: ExtractedMemory, new_date: str, candidates: list[Memory]) -> Decision:
        existing = "\n".join(
            f'<memory id="{m.id}" kind="{m.kind.value}" subject="{m.subject}" '
            f'date="{m.latest_date}">{m.statement}</memory>'
            for m in candidates
        )
        prompt = (
            f"<existing>\n{existing}\n</existing>\n\n"
            f'<new kind="{new.kind.value}" subject="{new.subject}" date="{new_date}">'
            f"{new.statement}</new>"
        )
        return self._parse(JUDGE_SYSTEM, prompt, Decision)

    def _parse(self, system: str, prompt: str, schema: type[T]) -> T:
        extra = {}
        if self.model in _FALLBACK_MODELS:
            # If a safety classifier declines, the API retries on a recommended model
            # inside the same call instead of returning a refusal.
            extra = {"betas": [_FALLBACK_BETA], "fallbacks": "default"}
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": self.effort},
            **extra,
        )
        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("Response was cut off at max_tokens; try a smaller chunk size.")
        if response.parsed_output is None:
            raise LLMError("The model returned no parseable output.")
        return response.parsed_output
