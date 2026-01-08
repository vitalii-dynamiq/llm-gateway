"""
Server-Sent Events (SSE) parser for streaming responses.

Handles:
- Standard SSE format (data:, event:, id:, retry:)
- Partial chunks and multi-byte UTF-8 boundaries
- OpenAI-style streaming format
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

__all__ = ["AsyncSSEParser", "SSEEvent", "SSEParser"]


@dataclass(slots=True)
class SSEEvent:
    """A single Server-Sent Event."""

    data: str
    event: str = "message"
    id: str | None = None
    retry: int | None = None

    @property
    def is_done(self) -> bool:
        """Check if this is the [DONE] terminator event."""
        return self.data.strip() == "[DONE]"


class SSEParser:
    """
    Parser for Server-Sent Events stream.

    Handles partial chunks and maintains state across calls.
    """

    __slots__ = ("_buffer", "_current_data", "_current_event", "_current_id", "_current_retry")

    def __init__(self) -> None:
        self._buffer = ""
        self._current_event = "message"
        self._current_data: list[str] = []
        self._current_id: str | None = None
        self._current_retry: int | None = None

    def feed(self, chunk: bytes | str) -> Iterator[SSEEvent]:
        """
        Feed a chunk of data and yield any complete events.

        Args:
            chunk: Bytes or string chunk from the stream

        Yields:
            SSEEvent for each complete event in the chunk
        """
        if isinstance(chunk, bytes):
            # Handle potential UTF-8 boundary issues
            chunk = chunk.decode("utf-8", errors="replace")

        self._buffer += chunk

        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")

            event = self._process_line(line)
            if event is not None:
                yield event

    def _process_line(self, line: str) -> SSEEvent | None:
        """Process a single line and return event if complete."""
        if not line:
            # Empty line = dispatch event
            if self._current_data:
                event = SSEEvent(
                    data="\n".join(self._current_data),
                    event=self._current_event,
                    id=self._current_id,
                    retry=self._current_retry,
                )
                # Reset for next event
                self._current_data = []
                self._current_event = "message"
                self._current_id = None
                self._current_retry = None
                return event
            return None

        # Comment line
        if line.startswith(":"):
            return None

        # Parse field
        if ":" in line:
            field, _, value = line.partition(":")
            # Remove single leading space if present
            if value.startswith(" "):
                value = value[1:]
        else:
            field = line
            value = ""

        if field == "data":
            self._current_data.append(value)
        elif field == "event":
            self._current_event = value
        elif field == "id":
            self._current_id = value
        elif field == "retry":
            with contextlib.suppress(ValueError):
                self._current_retry = int(value)

        return None

    def flush(self) -> SSEEvent | None:
        """Flush any remaining buffered event."""
        # First, process any remaining buffer content
        if self._buffer:
            self._process_line(self._buffer.rstrip("\r"))
            self._buffer = ""

        if self._current_data:
            event = SSEEvent(
                data="\n".join(self._current_data),
                event=self._current_event,
                id=self._current_id,
                retry=self._current_retry,
            )
            self._current_data = []
            self._current_event = "message"
            self._current_id = None
            self._current_retry = None
            return event
        return None


class AsyncSSEParser:
    """Async wrapper for SSE parsing from async byte streams."""

    __slots__ = ("_parser",)

    def __init__(self) -> None:
        self._parser = SSEParser()

    async def parse(self, stream: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
        """
        Parse SSE events from an async byte stream.

        Args:
            stream: Async iterator yielding bytes chunks

        Yields:
            SSEEvent for each complete event
        """
        async for chunk in stream:
            for event in self._parser.feed(chunk):
                yield event

        # Flush any remaining event
        final = self._parser.flush()
        if final:
            yield final


def parse_sse_stream(byte_stream: Iterator[bytes]) -> Iterator[SSEEvent]:
    """
    Convenience function to parse SSE events from a byte stream.

    Args:
        byte_stream: Iterator yielding bytes chunks

    Yields:
        SSEEvent for each complete event
    """
    parser = SSEParser()
    for chunk in byte_stream:
        yield from parser.feed(chunk)

    final = parser.flush()
    if final:
        yield final


async def parse_sse_stream_async(
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[SSEEvent]:
    """
    Convenience function to parse SSE events from an async byte stream.

    Args:
        byte_stream: Async iterator yielding bytes chunks

    Yields:
        SSEEvent for each complete event
    """
    parser = AsyncSSEParser()
    async for event in parser.parse(byte_stream):
        yield event
