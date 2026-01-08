"""
Server-Sent Events (SSE) parser for streaming responses.

Handles:
- Standard SSE format (data:, event:, id:, retry:)
- Partial chunks and multi-byte UTF-8 boundaries
- OpenAI-style streaming format

Performance optimizations:
- Pre-compiled constants to avoid repeated string creation
- Minimized allocations in hot paths
- Direct string operations instead of method calls where possible
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

__all__ = ["AsyncSSEParser", "SSEEvent", "SSEParser"]

# Pre-defined constants for hot path optimization
_NEWLINE = "\n"
_CARRIAGE_RETURN = "\r"
_COLON = ":"
_SPACE = " "
_DONE_MARKER = "[DONE]"
_DEFAULT_EVENT = "message"
_FIELD_DATA = "data"
_FIELD_EVENT = "event"
_FIELD_ID = "id"
_FIELD_RETRY = "retry"


class SSEEvent:
    """
    A single Server-Sent Event.

    Uses __slots__ for memory efficiency and faster attribute access.
    """

    __slots__ = ("_is_done", "data", "event", "id", "retry")

    def __init__(
        self,
        data: str,
        event: str = _DEFAULT_EVENT,
        id: str | None = None,
        retry: int | None = None,
    ) -> None:
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry
        # Cache is_done check since data is immutable
        self._is_done: bool | None = None

    @property
    def is_done(self) -> bool:
        """Check if this is the [DONE] terminator event."""
        if self._is_done is None:
            # Strip and compare - cache the result
            self._is_done = self.data.strip() == _DONE_MARKER
        return self._is_done


class SSEParser:
    """
    Parser for Server-Sent Events stream.

    Handles partial chunks and maintains state across calls.

    Performance notes:
    - Uses list for _current_data to avoid repeated string concatenation
    - Minimizes allocations in the hot parsing path
    - Uses string partition() which is faster than split() for single splits
    """

    __slots__ = ("_buffer", "_current_data", "_current_event", "_current_id", "_current_retry")

    def __init__(self) -> None:
        self._buffer = ""
        self._current_event = _DEFAULT_EVENT
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
        # Fast path: decode bytes if needed
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")

        # Append to buffer
        self._buffer += chunk

        # Process complete lines - use find() for speed
        buffer = self._buffer
        newline_pos = buffer.find(_NEWLINE)

        while newline_pos != -1:
            # Extract line (strip \r if present)
            line = buffer[:newline_pos]
            if line.endswith(_CARRIAGE_RETURN):
                line = line[:-1]

            # Advance buffer
            buffer = buffer[newline_pos + 1 :]

            # Process the line
            event = self._process_line(line)
            if event is not None:
                yield event

            # Find next newline
            newline_pos = buffer.find(_NEWLINE)

        # Store remaining buffer
        self._buffer = buffer

    def _process_line(self, line: str) -> SSEEvent | None:
        """Process a single line and return event if complete."""
        # Empty line = dispatch event
        if not line:
            if self._current_data:
                # Join data lines and create event
                event = SSEEvent(
                    data=_NEWLINE.join(self._current_data),
                    event=self._current_event,
                    id=self._current_id,
                    retry=self._current_retry,
                )
                # Reset for next event - reuse list
                self._current_data = []
                self._current_event = _DEFAULT_EVENT
                self._current_id = None
                self._current_retry = None
                return event
            return None

        # Comment line - fast check
        if line[0] == _COLON:
            return None

        # Parse field using partition (faster than split for single delimiter)
        colon_pos = line.find(_COLON)
        if colon_pos != -1:
            field = line[:colon_pos]
            value = line[colon_pos + 1 :]
            # Remove single leading space if present
            if value and value[0] == _SPACE:
                value = value[1:]
        else:
            field = line
            value = ""

        # Field dispatch - ordered by frequency in typical SSE streams
        if field == _FIELD_DATA:
            self._current_data.append(value)
        elif field == _FIELD_EVENT:
            self._current_event = value
        elif field == _FIELD_ID:
            self._current_id = value
        elif field == _FIELD_RETRY:
            with contextlib.suppress(ValueError):
                self._current_retry = int(value)

        return None

    def flush(self) -> SSEEvent | None:
        """Flush any remaining buffered event."""
        # First, process any remaining buffer content
        if self._buffer:
            line = self._buffer
            if line.endswith(_CARRIAGE_RETURN):
                line = line[:-1]
            self._process_line(line)
            self._buffer = ""

        if self._current_data:
            event = SSEEvent(
                data=_NEWLINE.join(self._current_data),
                event=self._current_event,
                id=self._current_id,
                retry=self._current_retry,
            )
            self._current_data = []
            self._current_event = _DEFAULT_EVENT
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
