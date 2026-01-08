#!/usr/bin/env python3
"""
Benchmarking script for fastlitellm.

Compares performance overhead of fastlitellm vs baseline (mocked HTTP).
Does NOT make real API calls - uses mocked responses.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

# Fixtures for mocked responses
MOCK_COMPLETION_RESPONSE = {
    "id": "chatcmpl-benchmark",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a benchmark test response with some content to parse.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}

MOCK_STREAMING_CHUNKS = [
    b'data: {"id":"1","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":"!"},"index":0}]}\n\n',
    b'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n',
    b"data: [DONE]\n\n",
]


def benchmark_import_time():
    """Benchmark import time."""
    import subprocess
    import sys

    # Time importing fastlitellm
    code = "import fastlitellm"
    result = subprocess.run(
        [sys.executable, "-c", f"import time; s=time.perf_counter(); {code}; print(time.perf_counter()-s)"],
        capture_output=True,
        text=True,
    )
    import_time = float(result.stdout.strip())
    print(f"Import time: {import_time*1000:.2f}ms")
    return import_time


def benchmark_type_creation(iterations: int = 10000):
    """Benchmark creating response types."""
    from fastlitellm.types import (
        Choice,
        Message,
        ModelResponse,
        Usage,
    )

    start = time.perf_counter()
    for _ in range(iterations):
        msg = Message(role="assistant", content="Hello world")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        ModelResponse(
            id="test-123",
            model="gpt-4o-mini",
            choices=[choice],
            usage=usage,
        )
    elapsed = time.perf_counter() - start

    print(f"Type creation ({iterations} iterations): {elapsed*1000:.2f}ms ({elapsed/iterations*1000000:.2f}µs/iter)")
    return elapsed


def benchmark_response_parsing(iterations: int = 10000):
    """Benchmark parsing JSON response."""
    from fastlitellm.providers.base import ProviderConfig
    from fastlitellm.providers.openai_adapter import OpenAIAdapter

    config = ProviderConfig(api_key="test-key")
    adapter = OpenAIAdapter(config)
    response_bytes = json.dumps(MOCK_COMPLETION_RESPONSE).encode("utf-8")

    start = time.perf_counter()
    for _ in range(iterations):
        adapter.parse_response(response_bytes, "gpt-4o-mini")
    elapsed = time.perf_counter() - start

    print(f"Response parsing ({iterations} iterations): {elapsed*1000:.2f}ms ({elapsed/iterations*1000000:.2f}µs/iter)")
    return elapsed


def benchmark_sse_parsing(iterations: int = 10000):
    """Benchmark SSE parsing."""
    from fastlitellm.http.sse import SSEParser

    # Create chunk data
    chunk_data = b"".join(MOCK_STREAMING_CHUNKS)

    start = time.perf_counter()
    for _ in range(iterations):
        parser = SSEParser()
        list(parser.feed(chunk_data))
    elapsed = time.perf_counter() - start

    print(f"SSE parsing ({iterations} iterations): {elapsed*1000:.2f}ms ({elapsed/iterations*1000000:.2f}µs/iter)")
    return elapsed


def benchmark_request_building(iterations: int = 10000):
    """Benchmark building requests."""
    from fastlitellm.providers.base import ProviderConfig
    from fastlitellm.providers.openai_adapter import OpenAIAdapter

    config = ProviderConfig(api_key="test-key")
    adapter = OpenAIAdapter(config)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]

    start = time.perf_counter()
    for _ in range(iterations):
        adapter.build_request(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
    elapsed = time.perf_counter() - start

    print(f"Request building ({iterations} iterations): {elapsed*1000:.2f}ms ({elapsed/iterations*1000000:.2f}µs/iter)")
    return elapsed


def benchmark_model_string_parsing(iterations: int = 100000):
    """Benchmark parsing model strings."""
    from fastlitellm.providers.base import parse_model_string

    test_strings = [
        "gpt-4o-mini",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-latest",
        "groq/llama-3.1-70b-versatile",
        "unknown-model",
    ]

    start = time.perf_counter()
    for _ in range(iterations):
        for s in test_strings:
            parse_model_string(s)
    elapsed = time.perf_counter() - start

    total_ops = iterations * len(test_strings)
    print(f"Model string parsing ({total_ops} ops): {elapsed*1000:.2f}ms ({elapsed/total_ops*1000000:.2f}µs/op)")
    return elapsed


def benchmark_cost_calculation(iterations: int = 100000):
    """Benchmark cost calculation."""
    from fastlitellm.pricing import cost_per_token

    start = time.perf_counter()
    for _ in range(iterations):
        cost_per_token("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    elapsed = time.perf_counter() - start

    print(f"Cost calculation ({iterations} iterations): {elapsed*1000:.2f}ms ({elapsed/iterations*1000000:.2f}µs/iter)")
    return elapsed


def benchmark_full_completion_mocked(iterations: int = 1000):
    """Benchmark full completion flow with mocked HTTP."""
    import fastlitellm
    from fastlitellm.http.client import HTTPResponse

    # Mock the HTTP client
    mock_response = HTTPResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body=json.dumps(MOCK_COMPLETION_RESPONSE).encode("utf-8"),
    )

    with patch("fastlitellm.core._get_http_client") as mock_client:
        mock_client.return_value.request.return_value = mock_response

        messages = [{"role": "user", "content": "Hello!"}]

        start = time.perf_counter()
        for _ in range(iterations):
            fastlitellm.completion(
                model="gpt-4o-mini",
                messages=messages,
                api_key="test-key",
            )
        elapsed = time.perf_counter() - start

    print(f"Full completion (mocked, {iterations} iterations): {elapsed*1000:.2f}ms ({elapsed/iterations*1000:.2f}ms/iter)")
    return elapsed


def benchmark_stream_chunk_builder(iterations: int = 1000):
    """Benchmark stream_chunk_builder."""
    from fastlitellm import stream_chunk_builder
    from fastlitellm.providers.base import ProviderConfig
    from fastlitellm.providers.openai_adapter import OpenAIAdapter

    config = ProviderConfig(api_key="test-key")
    adapter = OpenAIAdapter(config)

    # Parse mock streaming chunks
    chunks = []
    for chunk_data in MOCK_STREAMING_CHUNKS:
        data = chunk_data.decode("utf-8")
        if "data: " in data:
            json_str = data.replace("data: ", "").strip()
            if json_str and json_str != "[DONE]":
                chunk = adapter.parse_stream_event(json_str, "gpt-4o-mini")
                if chunk:
                    chunks.append(chunk)

    start = time.perf_counter()
    for _ in range(iterations):
        stream_chunk_builder(chunks)
    elapsed = time.perf_counter() - start

    print(f"Stream chunk builder ({iterations} iterations): {elapsed*1000:.2f}ms ({elapsed/iterations*1000:.2f}ms/iter)")
    return elapsed


def run_all_benchmarks():
    """Run all benchmarks."""
    print("=" * 60)
    print("fastlitellm Benchmarks")
    print("=" * 60)
    print()

    results = {}

    print("--- Startup ---")
    results["import"] = benchmark_import_time()
    print()

    print("--- Core Operations ---")
    results["type_creation"] = benchmark_type_creation()
    results["response_parsing"] = benchmark_response_parsing()
    results["request_building"] = benchmark_request_building()
    results["sse_parsing"] = benchmark_sse_parsing()
    print()

    print("--- Utility Operations ---")
    results["model_parsing"] = benchmark_model_string_parsing()
    results["cost_calculation"] = benchmark_cost_calculation()
    print()

    print("--- Integration (Mocked HTTP) ---")
    results["full_completion"] = benchmark_full_completion_mocked()
    results["stream_builder"] = benchmark_stream_chunk_builder()
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Import time: {results['import']*1000:.2f}ms")
    print(f"Parsing overhead per request: ~{results['response_parsing']/10000*1000000:.1f}µs")
    print(f"Request building overhead: ~{results['request_building']/10000*1000000:.1f}µs")
    print(f"Full completion overhead (mocked): ~{results['full_completion']/1000*1000:.2f}ms")


if __name__ == "__main__":
    run_all_benchmarks()
