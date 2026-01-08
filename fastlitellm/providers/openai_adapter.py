"""
OpenAI adapter for fastlitellm.

Supports:
- Chat completions (non-streaming and streaming)
- Tool/function calling
- Structured output (JSON mode and JSON schema)
- Embeddings
- Vision (image inputs)
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastlitellm.exceptions import (
    AuthenticationError,
    ContentFilterError,
    FastLiteLLMError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
    ResponseParseError,
    UnsupportedModelError,
)
from fastlitellm.providers.base import (
    COMMON_PARAMS,
    BaseAdapter,
    ProviderConfig,
    RequestData,
    register_provider,
)
from fastlitellm.types import (
    Choice,
    ChunkChoice,
    ChunkDelta,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    Message,
    ModelResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API."""

    provider_name = "openai"

    # OpenAI supports all common params plus some extras
    supported_params = COMMON_PARAMS | {
        "max_completion_tokens",  # For o1 models
        "logit_bias",
        "parallel_tool_calls",
        "service_tier",
        "store",
        "metadata",
        "stream_options",
        "reasoning_effort",  # For o1 models
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.openai.com/v1"

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("OPENAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build OpenAI chat completion request."""
        # Check params
        kwargs = self._check_params(drop_params, **kwargs)

        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Add stream_options for usage in streaming
        if stream:
            stream_options = kwargs.pop("stream_options", None)
            if stream_options:
                body["stream_options"] = stream_options

        # Handle max_tokens vs max_completion_tokens for o1 models
        if model.startswith("o1") and "max_tokens" in kwargs:
            # o1 models use max_completion_tokens
            if "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            else:
                kwargs.pop("max_tokens")

        # Add optional parameters
        optional_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "n",
            "logprobs",
            "top_logprobs",
            "user",
            "parallel_tool_calls",
            "service_tier",
            "store",
            "metadata",
            "reasoning_effort",
        ]

        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                body[param] = kwargs[param]

        # Handle tools
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                body["tool_choice"] = kwargs["tool_choice"]

        # Handle response_format
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]

        url = f"{self._api_base}/chat/completions"
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse OpenAI chat completion response."""
        try:
            resp = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse response JSON: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        return self._build_model_response(resp, model)

    def _build_model_response(self, resp: dict[str, Any], model: str) -> ModelResponse:
        """Build ModelResponse from parsed JSON."""
        choices: list[Choice] = []

        for choice_data in resp.get("choices", []):
            message_data = choice_data.get("message", {})

            # Parse tool calls if present
            tool_calls: list[ToolCall] | None = None
            if message_data.get("tool_calls"):
                tool_calls = []
                for tc in message_data["tool_calls"]:
                    func_data = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=FunctionCall(
                                name=func_data.get("name", ""),
                                arguments=func_data.get("arguments", ""),
                            ),
                        )
                    )

            # Parse legacy function_call if present
            function_call: FunctionCall | None = None
            if message_data.get("function_call"):
                fc = message_data["function_call"]
                function_call = FunctionCall(
                    name=fc.get("name", ""),
                    arguments=fc.get("arguments", ""),
                )

            message = Message(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                tool_calls=tool_calls,
                function_call=function_call,
                refusal=message_data.get("refusal"),
            )

            choices.append(
                Choice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                )
            )

        # Parse usage
        usage: Usage | None = None
        usage_data = resp.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
            )

        return ModelResponse(
            id=resp.get("id", ""),
            object=resp.get("object", "chat.completion"),
            created=resp.get("created", int(time.time())),
            model=resp.get("model", model),
            choices=choices,
            usage=usage,
            system_fingerprint=resp.get("system_fingerprint"),
            model_extra={"usage": usage.model_dump() if usage else {}},
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse OpenAI streaming event."""
        data = data.strip()
        if not data or data == "[DONE]":
            return None

        try:
            event = json.loads(data)
        except json.JSONDecodeError as e:
            raise ResponseParseError(
                f"Failed to parse stream event: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        choices: list[ChunkChoice] = []

        for choice_data in event.get("choices", []):
            delta_data = choice_data.get("delta", {})

            # Parse tool call deltas
            tool_calls: list[dict[str, Any]] | None = None
            if "tool_calls" in delta_data:
                tool_calls = delta_data["tool_calls"]

            delta = ChunkDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                tool_calls=tool_calls,
                function_call=delta_data.get("function_call"),
            )

            choices.append(
                ChunkChoice(
                    index=choice_data.get("index", 0),
                    delta=delta,
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                )
            )

        # Parse usage if present (with stream_options.include_usage=True)
        usage: Usage | None = None
        usage_data = event.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
            )

        return StreamChunk(
            id=event.get("id", ""),
            object=event.get("object", "chat.completion.chunk"),
            created=event.get("created", int(time.time())),
            model=event.get("model", model),
            choices=choices,
            usage=usage,
            system_fingerprint=event.get("system_fingerprint"),
        )

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> FastLiteLLMError:
        """Parse OpenAI error response."""
        try:
            error_data = json.loads(data.decode("utf-8"))
            error = error_data.get("error", {})
            message = error.get("message", "Unknown error")
            error_type = error.get("type", "")
            error_code = error.get("code", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = data.decode("utf-8", errors="replace")
            error_type = ""
            error_code = ""

        # Map to appropriate exception
        if status_code == 401:
            return AuthenticationError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        elif status_code == 429:
            return RateLimitError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        elif status_code == 400:
            if "content_filter" in error_code.lower() or "content_policy" in message.lower():
                return ContentFilterError(
                    message,
                    provider=self.provider_name,
                    status_code=status_code,
                    request_id=request_id,
                    filter_reason=error_code,
                )
            return InvalidRequestError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        elif status_code == 404:
            return UnsupportedModelError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        else:
            return ProviderAPIError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
                error_type=error_type,
                error_code=error_code,
            )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build OpenAI embedding request."""
        body: dict[str, Any] = {
            "model": model,
            "input": input,
        }

        # Optional parameters
        if "encoding_format" in kwargs:
            body["encoding_format"] = kwargs["encoding_format"]
        if "dimensions" in kwargs:
            body["dimensions"] = kwargs["dimensions"]
        if "user" in kwargs:
            body["user"] = kwargs["user"]

        url = f"{self._api_base}/embeddings"
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Parse OpenAI embedding response."""
        try:
            resp = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse embedding response: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        embeddings: list[EmbeddingData] = []
        for item in resp.get("data", []):
            embeddings.append(
                EmbeddingData(
                    index=item.get("index", 0),
                    embedding=item.get("embedding", []),
                    object=item.get("object", "embedding"),
                )
            )

        usage_data = resp.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return EmbeddingResponse(
            model=resp.get("model", model),
            data=embeddings,
            usage=usage,
            object=resp.get("object", "list"),
        )


# Register on import
register_provider("openai", OpenAIAdapter)
