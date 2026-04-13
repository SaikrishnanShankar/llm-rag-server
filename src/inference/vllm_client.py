"""
vLLM client — OpenAI-compatible, works with real vLLM or any OpenAI endpoint.

Design decision: single factory function `get_llm_client()` decides at
startup whether to return the real OpenAI client (pointing at vLLM) or
the mock client based on VLLM_BASE_URL.  All agent nodes call this
factory — swapping mock → real on PACE is a one-env-var change.

The `generate` and `generate_stream` functions are the public API used
by the LangGraph nodes; they abstract over the underlying client type.
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

from src.config import settings
from src.inference.mock_client import MockLLMClient

logger = logging.getLogger(__name__)

_client = None


def get_llm_client():
    """Return a cached LLM client (real or mock) based on VLLM_BASE_URL."""
    global _client
    if _client is not None:
        return _client

    if settings.vllm_base_url.startswith("mock://"):
        logger.info("Using mock LLM client (VLLM_BASE_URL=%s)", settings.vllm_base_url)
        _client = MockLLMClient(model=settings.vllm_model)
    else:
        try:
            from openai import AsyncOpenAI
            logger.info("Connecting to vLLM at %s", settings.vllm_base_url)
            _client = AsyncOpenAI(
                base_url=settings.vllm_base_url,
                api_key="EMPTY",  # vLLM doesn't require a real key
            )
        except ImportError:
            logger.warning("openai package not available, falling back to mock client")
            _client = MockLLMClient(model=settings.vllm_model)

    return _client


async def generate(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """
    Generate a completion and return (answer_text, usage_stats).

    Args:
        messages:    OpenAI chat format: [{"role": "...", "content": "..."}]
        temperature: Overrides settings.vllm_temperature if provided.
        max_tokens:  Overrides settings.vllm_max_tokens if provided.

    Returns:
        Tuple of (answer_text, usage_dict) where usage_dict has keys:
        prompt_tokens, completion_tokens, total_tokens, latency_seconds.
    """
    client = get_llm_client()
    t0 = time.perf_counter()

    response = await client.chat.completions.create(
        model=settings.vllm_model,
        messages=messages,
        temperature=temperature if temperature is not None else settings.vllm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.vllm_max_tokens,
        stream=False,
    )

    latency = time.perf_counter() - t0
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_seconds": latency,
        "tokens_per_second": response.usage.completion_tokens / max(latency, 1e-6),
    }
    return content, usage


async def generate_stream(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    """
    Streaming generation — yields text chunks as they are produced.

    Usage:
        async for chunk in generate_stream(messages):
            yield chunk
    """
    client = get_llm_client()

    stream = await client.chat.completions.create(
        model=settings.vllm_model,
        messages=messages,
        temperature=temperature if temperature is not None else settings.vllm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.vllm_max_tokens,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def build_rag_messages(
    query: str,
    context_chunks: List[Dict],
    system_prefix: str = "",
) -> List[Dict[str, str]]:
    """
    Construct the OpenAI messages list for a RAG query.

    The context is injected into the system message so the model always
    sees it, regardless of conversation history length.
    """
    context_text = "\n\n".join(
        f"[Source {i+1}: {c.get('doc_id', 'unknown')}]\n{c['content']}"
        for i, c in enumerate(context_chunks)
    )

    system_content = f"""{system_prefix}
You are a helpful assistant that answers questions based on the provided context.
Always ground your answers in the context passages below.
If the context does not contain enough information, say so clearly.

Context passages:
{context_text}

Instructions:
- Answer the question directly and concisely.
- Cite sources using [Source N] notation when referencing specific information.
- Do not fabricate information beyond what is in the context.
""".strip()

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]


def build_direct_messages(query: str) -> List[Dict[str, str]]:
    """Messages for queries that don't require retrieval (factual/general)."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions clearly and concisely.",
        },
        {"role": "user", "content": query},
    ]
