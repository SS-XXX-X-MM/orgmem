"""Exercise AnthropicLLM through the real SDK with a mocked HTTP transport (no network)."""

import json

import anthropic
import httpx2
import pytest

from orgmem.llm import AnthropicLLM, LLMError


def make_client(reply: dict, stop_reason: str = "end_turn", captured: list | None = None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if captured is not None:
            captured.append((dict(request.headers), json.loads(request.content)))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": json.dumps(reply)}],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 10},
            },
        )

    return anthropic.Anthropic(
        api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )


def test_extract_sends_schema_effort_and_fallbacks():
    captured = []
    reply = {
        "memories": [
            {
                "kind": "decision",
                "subject": "billing",
                "statement": "Billing uses Postgres.",
                "evidence": "billing service will use Postgres",
            }
        ]
    }
    llm = AnthropicLLM(client=make_client(reply, captured=captured))
    memories = llm.extract("text", "notes.md", "2026-03-02")

    assert memories[0].statement == "Billing uses Postgres."
    headers, body = captured[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers["anthropic-beta"]


def test_no_fallbacks_for_models_that_do_not_support_them():
    captured = []
    llm = AnthropicLLM(
        model="claude-haiku-5-5", client=make_client({"memories": []}, captured=captured)
    )
    llm.extract("text", "notes.md", "2026-03-02")
    assert "fallbacks" not in captured[0][1]


def test_refusal_raises():
    llm = AnthropicLLM(client=make_client({"memories": []}, stop_reason="refusal"))
    with pytest.raises(LLMError):
        llm.extract("text", "notes.md", "2026-03-02")
