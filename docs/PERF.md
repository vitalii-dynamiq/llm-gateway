# Performance Guide

This document explains fastlitellm's performance characteristics, optimization strategies, and how to maintain performance over time.

## Design Principles

fastlitellm is designed for **minimal overhead** on top of network latency:

1. **Zero runtime dependencies** - No third-party packages at runtime
2. **Minimal allocations** - Use `__slots__`, pre-allocated structures
3. **Fast parsing** - Optimized SSE parser, direct JSON handling
4. **Lazy initialization** - HTTP clients created on first use
5. **Connection reuse** - Connection pooling for HTTP/1.1

## Benchmark Results

Typical performance on modern hardware (measured on GitHub Actions runners):

| Operation | Target | Typical |
|-----------|--------|---------|
| Import time | < 500ms | ~150-200ms |
| Type creation | < 50µs | ~10-15µs |
| Response parsing | < 100µs | ~30-50µs |
| Request building | < 100µs | ~20-40µs |
| SSE parsing (per stream) | < 50µs | ~15-25µs |
| Model string parsing | < 5µs | ~1-2µs |
| Full completion (mocked) | < 5ms | ~1-2ms |
| Stream chunk builder | < 1ms | ~0.1-0.3ms |

### Streaming Latency

| Metric | Target | Typical |
|--------|--------|---------|
| Time-to-first-token (TTFT) | < 100µs | ~30-50µs |
| Per-chunk overhead | < 20µs | ~5-10µs |

## Hot Path Optimizations

### 1. SSE Parser

The SSE parser is heavily optimized since it runs for every streaming chunk:

```python
# Pre-defined constants avoid repeated string creation
_NEWLINE = "\n"
_COLON = ":"
_FIELD_DATA = "data"

# Use find() instead of split() for single-delimiter parsing
newline_pos = buffer.find(_NEWLINE)

# Cache is_done check since event data is immutable
@property
def is_done(self) -> bool:
    if self._is_done is None:
        self._is_done = self.data.strip() == "[DONE]"
    return self._is_done
```

### 2. Type Creation

All response types use `__slots__` for memory efficiency:

```python
@dataclass(slots=True)
class Message:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
```

### 3. Stream Chunk Builder

Optimized to minimize dict operations and string concatenations:

```python
# Use lists for accumulation, join once at the end
choice_content: dict[int, list[str]] = {}

# Tool calls stored as [id, type, name_parts, arg_parts]
# to avoid repeated dict creation
tc_dict[tc_index] = ["", "function", [], []]
```

### 4. JSON Handling

Uses stdlib `json` module which is implemented in C:

```python
# Direct encoding without intermediate structures
body = json.dumps(data, ensure_ascii=False).encode("utf-8")
```

## Running Benchmarks

### Quick benchmark

```bash
python benchmarks/benchmark.py
```

### CI regression tests

```bash
python benchmarks/benchmark_ci.py
```

### Profile a specific operation

```python
import cProfile
import pstats

from fastlitellm.http.sse import SSEParser

def profile_sse():
    parser = SSEParser()
    data = b"data: test\n\n" * 1000
    for _ in range(1000):
        list(parser.feed(data))

cProfile.run('profile_sse()', 'sse.prof')
stats = pstats.Stats('sse.prof')
stats.sort_stats('cumulative').print_stats(20)
```

## Keeping It Fast

### When Adding Features

1. **Benchmark before and after** - Run `benchmark_ci.py`
2. **Avoid hot-path allocations** - No new objects per request/chunk
3. **Use `__slots__`** - For all dataclasses in types.py
4. **Pre-compute constants** - Move string literals outside loops
5. **Prefer stdlib** - C-implemented stdlib is faster than pure Python

### Code Review Checklist

- [ ] No new runtime dependencies added
- [ ] Types use `slots=True`
- [ ] No allocations in SSE parser feed loop
- [ ] No repeated dict/list creation per chunk
- [ ] Constants defined at module level
- [ ] Benchmark regression tests pass

### Known Performance Traps

1. **String concatenation in loops** - Use list + join instead
2. **Repeated dict.get()** - Cache in local variable
3. **Attribute access in tight loops** - Cache `obj.attr` in local var
4. **Creating new lists/dicts per iteration** - Pre-allocate or reuse
5. **Using `split()` when `partition()` or `find()` works** - Latter are faster

## Memory Usage

fastlitellm is designed to be memory-efficient:

| Component | Memory Strategy |
|-----------|-----------------|
| Response types | `__slots__` reduces per-instance overhead |
| SSE parser | Bounded buffer, no unbounded accumulation |
| Stream chunks | Yielded immediately, not accumulated |
| Connection pool | Configurable max connections (default: 10) |

### Streaming Memory Behavior

```
Request start
    │
    ▼
┌─────────────────────┐
│ SSE Parser Buffer   │  ← Bounded by chunk size (~8KB)
└─────────────────────┘
    │
    ▼ (yield immediately)
┌─────────────────────┐
│ StreamChunk object  │  ← Temporary, GC'd after yield
└─────────────────────┘
    │
    ▼
User consumes chunk
```

The parser never accumulates unbounded data - each event is yielded as soon as complete.

## CI Performance Gates

The CI pipeline includes automatic performance regression detection:

```yaml
# .github/workflows/ci.yml
- name: Run benchmark regression tests
  run: python benchmarks/benchmark_ci.py
```

Thresholds are defined in `benchmarks/benchmark_ci.py`:

```python
MAX_IMPORT_TIME_MS = 500      # Import time
MAX_TYPE_CREATION_US = 50     # Per-type creation
MAX_RESPONSE_PARSING_US = 100 # Per-response parsing
MAX_SSE_PARSING_US = 50       # Per-stream parsing
MAX_FULL_COMPLETION_MS = 5    # Full request (mocked)
```

If any threshold is exceeded, the CI job fails.

## Comparing to LiteLLM

fastlitellm aims for significantly lower overhead than LiteLLM:

| Metric | fastlitellm | LiteLLM |
|--------|-------------|---------|
| Import time | ~200ms | ~2-5s |
| Runtime deps | 0 | 50+ |
| Memory footprint | ~10MB | ~100MB+ |
| Cold start | ~200ms | ~5s+ |

The difference is most noticeable in:
- Serverless/Lambda cold starts
- CLI tools
- High-throughput streaming applications

## Future Optimizations

Potential areas for further optimization (when needed):

1. **orjson integration** - Optional dependency for faster JSON (when stdlib json is bottleneck)
2. **uvloop integration** - Optional for async workloads
3. **Response caching** - For repeated identical requests
4. **HTTP/2 support** - Multiplexing for high-concurrency scenarios

These are intentionally not implemented to maintain the "zero dependencies" principle, but could be added as optional extras.

## Measuring Your Application

For production monitoring:

```python
import time
from fastlitellm import completion

start = time.perf_counter()
response = completion(model="gpt-4o-mini", messages=[...])
latency = time.perf_counter() - start

# Most of latency is network time, not library overhead
# Library overhead should be < 5ms
print(f"Total latency: {latency*1000:.2f}ms")
```

For streaming, measure time-to-first-token:

```python
start = time.perf_counter()
response = completion(model="gpt-4o-mini", messages=[...], stream=True)

for i, chunk in enumerate(response):
    if i == 0:
        ttft = time.perf_counter() - start
        print(f"Time to first token: {ttft*1000:.2f}ms")
    # process chunk
```
