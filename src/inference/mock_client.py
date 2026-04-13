"""
Mock vLLM client for local development — no GPU required.

Design decision: the mock returns deterministic, context-grounded answers
rather than Lorem Ipsum. It extracts key sentences from the retrieved
context so RAGAS faithfulness scores are real (the answer IS grounded in
context), enabling meaningful eval runs locally before PACE deployment.
"""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, List


class MockCompletionResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int, model: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )()
        self.model = model


class MockLLMClient:
    """
    Mimics the OpenAI client interface used by VLLMClient.

    Returns answers synthesised from the context passages so that RAGAS
    faithfulness evaluation produces non-trivial scores.
    """

    def __init__(self, model: str = "mock-llama"):
        self.model = model
        self.chat = _ChatCompletions(model)

    @property
    def is_mock(self) -> bool:
        return True


class _ChatCompletions:
    def __init__(self, model: str):
        self.model = model
        self.completions = self  # openai-style: client.chat.completions.create(...)

    async def create(
        self,
        messages: List[dict],
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 512,
        stream: bool = False,
    ):
        model = model or self.model
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")

        answer = _synthesise_answer(user_message, system_message)
        prompt_tokens = len(" ".join(m["content"] for m in messages).split())
        completion_tokens = len(answer.split())

        # Simulate inference latency proportional to output length
        await asyncio.sleep(min(0.05 * completion_tokens / 10, 0.5))

        if stream:
            return _stream_response(answer, model)

        return MockCompletionResponse(answer, prompt_tokens, completion_tokens, model)


async def _stream_response(answer: str, model: str) -> AsyncIterator[object]:
    words = answer.split()
    for i, word in enumerate(words):
        chunk_text = word + (" " if i < len(words) - 1 else "")
        await asyncio.sleep(0.01)
        yield type(
            "Chunk",
            (),
            {
                "choices": [
                    type(
                        "Ch",
                        (),
                        {
                            "delta": type("D", (), {"content": chunk_text})(),
                            "finish_reason": "stop" if i == len(words) - 1 else None,
                        },
                    )()
                ],
                "model": model,
            },
        )()


def _synthesise_answer(user_message: str, system_message: str) -> str:
    """
    Build a grounded answer by extracting relevant sentences from the
    context block embedded in the system prompt.
    """
    # Extract context passages from the system prompt
    context_match = re.search(r"Context passages:(.*?)(?:Instructions:|$)", system_message, re.DOTALL)
    if not context_match:
        return _fallback_answer(user_message)

    context = context_match.group(1).strip()

    # Extract question keywords (simple heuristic)
    question_words = set(re.findall(r"\b\w{4,}\b", user_message.lower()))
    stop_words = {"what", "when", "where", "which", "how", "does", "this", "that", "with", "from", "about", "explain", "describe"}
    keywords = question_words - stop_words

    # Score sentences by keyword overlap
    sentences = re.split(r"(?<=[.!?])\s+", context)
    scored = []
    for sent in sentences:
        sent_words = set(re.findall(r"\b\w{4,}\b", sent.lower()))
        score = len(keywords & sent_words)
        scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    top_sentences = [s for _, s in scored[:4] if s.strip()]

    if not top_sentences:
        return _fallback_answer(user_message)

    answer = " ".join(top_sentences)
    # Add a natural intro
    intro = f"Based on the available information: {answer}"
    return intro.strip()


def _fallback_answer(query: str) -> str:
    return (
        f"This is a mock response for the query: '{query}'. "
        "In production, this would be answered by a real LLM (e.g., Llama on vLLM). "
        "Set VLLM_BASE_URL to your vLLM server endpoint to enable real inference."
    )
