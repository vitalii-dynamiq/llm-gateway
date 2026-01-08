# AGENTS.md - Guide for AI Coding Agents

This document provides guidelines for AI coding agents working on the arcllm codebase.

## Project Overview

arcllm is a lightweight, high-performance Python library for calling LLM providers. It's designed to be:
- API-compatible with LiteLLM SDK
- Zero runtime dependencies (stdlib only)
- Fast and efficient
- Easy to maintain by autonomous agents

## Code Style & Conventions

### Python Style
- **Target Python 3.13+** - Use modern typing syntax
- Use `from __future__ import annotations` in all files
- Type hints are required for all public functions
- Use `__slots__` for dataclasses when possible
- Follow ruff/pyright strict mode

### Naming Conventions
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`

### Import Style
```python
from __future__ import annotations

import json
import os
from typing import Any

from arcllm.types import ModelResponse
from arcllm.exceptions import ArcLLMError
```

## Architecture

```
arcllm/
├── __init__.py          # Public API exports
├── types.py             # All dataclasses (ModelResponse, etc.)
├── exceptions.py        # Exception classes
├── core.py              # Main API functions (completion, etc.)
├── http/
│   ├── client.py        # Sync HTTP client
│   ├── async_client.py  # Async HTTP client
│   └── sse.py           # SSE parser
├── providers/
│   ├── base.py          # Adapter protocol & registry
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   └── ...              # One file per provider
├── pricing/
│   └── tables.py        # Pricing data
└── capabilities/
    └── tables.py        # Model capabilities
```

## Key Invariants

### 1. Response Structure
All responses must have this structure:
```python
response.choices[0].message.content  # Always accessible
response.choices[0].message.tool_calls  # List[ToolCall] or None
response.model_extra["usage"]  # Dict with token counts
```

### 2. Tool Calls Format
Tool calls must follow OpenAI format:
```python
tool_call.id = "call_xxx"
tool_call.type = "function"
tool_call.function.name = "function_name"
tool_call.function.arguments = '{"json": "string"}'  # JSON STRING
```

### 3. Usage Tracking
- Always use provider-reported usage
- Never count tokens ourselves
- Usage can be None if provider doesn't report it

### 4. Error Handling
- Map all provider errors to arcllm exception classes
- Include provider name in all exceptions
- Preserve request_id when available

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=arcllm

# Run specific test file
pytest tests/test_types.py

# Run type checking
mypy arcllm

# Run linting
ruff check arcllm
ruff format arcllm
```

## Adding/Updating Features

### Adding a New Provider
See `docs/ADDING_A_PROVIDER.md` for detailed instructions.

Quick checklist:
1. Create `providers/newprovider_adapter.py`
2. Implement Adapter protocol
3. Register in `providers/base.py`
4. Add pricing to `pricing/tables.py`
5. Add capabilities to `capabilities/tables.py`
6. Write tests in `tests/providers/`

### Updating Pricing
1. Edit `arcllm/pricing/tables.py`
2. Update `PRICING_VERSION` at top of file
3. Run tests: `pytest tests/test_pricing.py`

### Updating Capabilities
1. Edit `arcllm/capabilities/tables.py`
2. Update `CAPABILITIES_VERSION` at top of file
3. Run tests: `pytest tests/test_capabilities.py`

### Adding a New Parameter
1. Add to `COMMON_PARAMS` in `providers/base.py` if common
2. Or add to specific adapter's `supported_params`
3. Handle in `build_request()` method
4. Add tests

## Common Tasks

### Fix a Bug
1. Write a failing test first
2. Fix the bug
3. Verify test passes
4. Run full test suite

### Add Tests
- Unit tests go in `tests/`
- Provider tests go in `tests/providers/`
- Use fixtures from `tests/conftest.py`
- Test both success and error cases

### Update Dependencies
- Runtime: NO new dependencies (stdlib only)
- Dev: Update in `pyproject.toml [project.optional-dependencies]`

## Do's and Don'ts

### DO
- Keep modules small and focused
- Use explicit types everywhere
- Test edge cases
- Preserve backwards compatibility
- Document public APIs

### DON'T
- Add runtime dependencies
- Count tokens ourselves
- Block the event loop in async code
- Swallow exceptions silently
- Break the response interface contract

## Common Patterns

### Provider Adapter Pattern
```python
class NewAdapter(BaseAdapter):
    provider_name = "newprovider"
    
    def build_request(self, *, model, messages, **kwargs) -> RequestData:
        # Convert to provider format
        body = self._build_body(model, messages, **kwargs)
        return RequestData(method="POST", url=url, headers=headers, body=body)
    
    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        # Parse provider response
        resp = json.loads(data)
        return self._build_model_response(resp, model)
```

### Error Mapping Pattern
```python
def parse_error(self, status_code, data, request_id):
    message = self._extract_error_message(data)
    
    if status_code == 401:
        return AuthenticationError(message, provider=self.provider_name)
    elif status_code == 429:
        return RateLimitError(message, provider=self.provider_name)
    # ... etc
```

### Streaming Pattern
```python
def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
    if not data or data == "[DONE]":
        return None
    
    event = json.loads(data)
    return StreamChunk(
        id=event.get("id", ""),
        choices=[self._parse_chunk_choice(c) for c in event.get("choices", [])],
    )
```

## Debugging Tips

1. **Request Issues**: Print `request.body.decode()` to see what's being sent
2. **Response Issues**: Print raw bytes before parsing
3. **Streaming Issues**: Check SSE parser output with test data
4. **Auth Issues**: Verify env vars are set correctly

## Performance Considerations

- Minimize object allocations in hot paths
- Use `__slots__` for dataclasses
- Avoid unnecessary JSON parsing
- Reuse HTTP connections when possible
- Don't import heavy modules at startup
