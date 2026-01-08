#!/usr/bin/env python3
"""
Benchmark regression tests for CI.

These tests ensure performance doesn't regress beyond acceptable thresholds.
All thresholds are generous (2-3x baseline) to avoid flaky CI failures
while still catching major regressions.

No network calls - all tests use mocked HTTP responses.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

# =============================================================================
# Threshold Configuration (generous to avoid flaky CI)
# =============================================================================

# Import time should be under 500ms (generous for cold start)
MAX_IMPORT_TIME_MS = 500

# Type creation: max 50µs per operation
MAX_TYPE_CREATION_US = 50

# JSON parsing: max 100µs per operation (includes type instantiation)
MAX_RESPONSE_PARSING_US = 100

# Request building: max 100µs per operation
MAX_REQUEST_BUILDING_US = 100

# SSE parsing: max 50µs per stream
MAX_SSE_PARSING_US = 50

# Model string parsing: max 5µs per operation
MAX_MODEL_PARSING_US = 5

# Full completion (mocked): max 5ms per operation
MAX_FULL_COMPLETION_MS = 5

# Stream chunk builder: max 1ms per operation
MAX_STREAM_BUILDER_MS = 1


# =============================================================================
# Test Fixtures
# =============================================================================

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
                "content": "This is a benchmark test response.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}

MOCK_STREAMING_CHUNKS = [
    b'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"gpt-4o-mini","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"gpt-4o-mini","choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"gpt-4o-mini","choices":[{"delta":{"content":" world"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"gpt-4o-mini","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n',
    b"data: [DONE]\n\n",
]


# =============================================================================
# Benchmark Functions
# =============================================================================


def measure_import_time() -> float:
    """Measure import time in milliseconds."""
    code = "import arcllm"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import time; s=time.perf_counter(); {code}; print(time.perf_counter()-s)",
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip()) * 1000  # Convert to ms


def measure_type_creation(iterations: int = 5000) -> float:
    """Measure type creation time in microseconds per operation."""
    from arcllm.types import Choice, Message, ModelResponse, Usage

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
    return (elapsed / iterations) * 1_000_000  # Convert to µs


def measure_response_parsing(iterations: int = 5000) -> float:
    """Measure response parsing time in microseconds per operation."""
    from arcllm.providers.base import ProviderConfig
    from arcllm.providers.openai_adapter import OpenAIAdapter

    config = ProviderConfig(api_key="test-key")
    adapter = OpenAIAdapter(config)
    response_bytes = json.dumps(MOCK_COMPLETION_RESPONSE).encode("utf-8")

    start = time.perf_counter()
    for _ in range(iterations):
        adapter.parse_response(response_bytes, "gpt-4o-mini")
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1_000_000  # Convert to µs


def measure_request_building(iterations: int = 5000) -> float:
    """Measure request building time in microseconds per operation."""
    from arcllm.providers.base import ProviderConfig
    from arcllm.providers.openai_adapter import OpenAIAdapter

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
    return (elapsed / iterations) * 1_000_000  # Convert to µs


def measure_sse_parsing(iterations: int = 5000) -> float:
    """Measure SSE parsing time in microseconds per stream."""
    from arcllm.http.sse import SSEParser

    chunk_data = b"".join(MOCK_STREAMING_CHUNKS)

    start = time.perf_counter()
    for _ in range(iterations):
        parser = SSEParser()
        list(parser.feed(chunk_data))
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1_000_000  # Convert to µs


def measure_model_parsing(iterations: int = 50000) -> float:
    """Measure model string parsing time in microseconds per operation."""
    from arcllm.providers.base import parse_model_string

    test_strings = [
        "gpt-4o-mini",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-latest",
    ]

    start = time.perf_counter()
    for _ in range(iterations):
        for s in test_strings:
            parse_model_string(s)
    elapsed = time.perf_counter() - start
    total_ops = iterations * len(test_strings)
    return (elapsed / total_ops) * 1_000_000  # Convert to µs


def measure_full_completion_mocked(iterations: int = 500) -> float:
    """Measure full completion flow with mocked HTTP in milliseconds."""
    from unittest.mock import patch

    import arcllm
    from arcllm.http.client import HTTPResponse

    mock_response = HTTPResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body=json.dumps(MOCK_COMPLETION_RESPONSE).encode("utf-8"),
    )

    with patch("arcllm.core._get_http_client") as mock_client:
        mock_client.return_value.request.return_value = mock_response

        messages = [{"role": "user", "content": "Hello!"}]

        start = time.perf_counter()
        for _ in range(iterations):
            arcllm.completion(
                model="gpt-4o-mini",
                messages=messages,
                api_key="test-key",
            )
        elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # Convert to ms


def measure_stream_chunk_builder(iterations: int = 1000) -> float:
    """Measure stream chunk builder time in milliseconds."""
    from arcllm import stream_chunk_builder
    from arcllm.providers.base import ProviderConfig
    from arcllm.providers.openai_adapter import OpenAIAdapter

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
    return (elapsed / iterations) * 1000  # Convert to ms


# =============================================================================
# CI Test Runner
# =============================================================================


def run_benchmark_tests() -> bool:
    """Run all benchmark tests and check against thresholds."""
    all_passed = True
    results: dict[str, dict[str, Any]] = {}

    print("=" * 70)
    print("Performance Benchmark Regression Tests")
    print("=" * 70)
    print()

    # Import time
    print("Testing import time...", end=" ", flush=True)
    import_time = measure_import_time()
    passed = import_time <= MAX_IMPORT_TIME_MS
    results["import_time"] = {
        "value": import_time,
        "unit": "ms",
        "threshold": MAX_IMPORT_TIME_MS,
        "passed": passed,
    }
    print(f"{import_time:.2f}ms (threshold: {MAX_IMPORT_TIME_MS}ms) {'✓' if passed else '✗'}")
    all_passed &= passed

    # Type creation
    print("Testing type creation...", end=" ", flush=True)
    type_time = measure_type_creation()
    passed = type_time <= MAX_TYPE_CREATION_US
    results["type_creation"] = {
        "value": type_time,
        "unit": "µs",
        "threshold": MAX_TYPE_CREATION_US,
        "passed": passed,
    }
    print(f"{type_time:.2f}µs/op (threshold: {MAX_TYPE_CREATION_US}µs) {'✓' if passed else '✗'}")
    all_passed &= passed

    # Response parsing
    print("Testing response parsing...", end=" ", flush=True)
    parse_time = measure_response_parsing()
    passed = parse_time <= MAX_RESPONSE_PARSING_US
    results["response_parsing"] = {
        "value": parse_time,
        "unit": "µs",
        "threshold": MAX_RESPONSE_PARSING_US,
        "passed": passed,
    }
    print(
        f"{parse_time:.2f}µs/op (threshold: {MAX_RESPONSE_PARSING_US}µs) {'✓' if passed else '✗'}"
    )
    all_passed &= passed

    # Request building
    print("Testing request building...", end=" ", flush=True)
    build_time = measure_request_building()
    passed = build_time <= MAX_REQUEST_BUILDING_US
    results["request_building"] = {
        "value": build_time,
        "unit": "µs",
        "threshold": MAX_REQUEST_BUILDING_US,
        "passed": passed,
    }
    print(
        f"{build_time:.2f}µs/op (threshold: {MAX_REQUEST_BUILDING_US}µs) {'✓' if passed else '✗'}"
    )
    all_passed &= passed

    # SSE parsing
    print("Testing SSE parsing...", end=" ", flush=True)
    sse_time = measure_sse_parsing()
    passed = sse_time <= MAX_SSE_PARSING_US
    results["sse_parsing"] = {
        "value": sse_time,
        "unit": "µs",
        "threshold": MAX_SSE_PARSING_US,
        "passed": passed,
    }
    print(f"{sse_time:.2f}µs/stream (threshold: {MAX_SSE_PARSING_US}µs) {'✓' if passed else '✗'}")
    all_passed &= passed

    # Model parsing
    print("Testing model string parsing...", end=" ", flush=True)
    model_time = measure_model_parsing()
    passed = model_time <= MAX_MODEL_PARSING_US
    results["model_parsing"] = {
        "value": model_time,
        "unit": "µs",
        "threshold": MAX_MODEL_PARSING_US,
        "passed": passed,
    }
    print(f"{model_time:.2f}µs/op (threshold: {MAX_MODEL_PARSING_US}µs) {'✓' if passed else '✗'}")
    all_passed &= passed

    # Full completion (mocked)
    print("Testing full completion (mocked)...", end=" ", flush=True)
    completion_time = measure_full_completion_mocked()
    passed = completion_time <= MAX_FULL_COMPLETION_MS
    results["full_completion"] = {
        "value": completion_time,
        "unit": "ms",
        "threshold": MAX_FULL_COMPLETION_MS,
        "passed": passed,
    }
    print(
        f"{completion_time:.2f}ms/req (threshold: {MAX_FULL_COMPLETION_MS}ms) {'✓' if passed else '✗'}"
    )
    all_passed &= passed

    # Stream chunk builder
    print("Testing stream chunk builder...", end=" ", flush=True)
    builder_time = measure_stream_chunk_builder()
    passed = builder_time <= MAX_STREAM_BUILDER_MS
    results["stream_builder"] = {
        "value": builder_time,
        "unit": "ms",
        "threshold": MAX_STREAM_BUILDER_MS,
        "passed": passed,
    }
    print(
        f"{builder_time:.3f}ms/call (threshold: {MAX_STREAM_BUILDER_MS}ms) {'✓' if passed else '✗'}"
    )
    all_passed &= passed

    print()
    print("=" * 70)
    if all_passed:
        print("All benchmark tests PASSED ✓")
    else:
        print("Some benchmark tests FAILED ✗")
        print()
        print("Failed tests:")
        for name, data in results.items():
            if not data["passed"]:
                print(
                    f"  - {name}: {data['value']:.2f}{data['unit']} "
                    f"(threshold: {data['threshold']}{data['unit']})"
                )
    print("=" * 70)

    # Write results to JSON for artifact collection
    with open("benchmark-ci-results.json", "w") as f:
        json.dump(results, f, indent=2)

    return all_passed


if __name__ == "__main__":
    success = run_benchmark_tests()
    sys.exit(0 if success else 1)
