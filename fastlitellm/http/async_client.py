"""
Asynchronous HTTP client using stdlib asyncio.

Features:
- True async I/O (no blocking)
- TLS support
- Timeout handling
- Streaming response support
- Retries with exponential backoff and jitter
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import gzip
import json
import random
import ssl
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fastlitellm.exceptions import (
    ConnectionError,
    ProviderAPIError,
    TimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["AsyncHTTPClient", "AsyncHTTPResponse"]


@dataclass(slots=True)
class AsyncHTTPResponse:
    """Response from an async HTTP request."""

    status_code: int
    headers: dict[str, str]
    body: bytes
    request_id: str | None = None

    def json(self) -> Any:
        """Parse response body as JSON."""
        return json.loads(self.body.decode("utf-8"))

    @property
    def text(self) -> str:
        """Return response body as text."""
        return self.body.decode("utf-8")


class AsyncHTTPClient:
    """
    Asynchronous HTTP client using asyncio streams.

    Uses low-level asyncio.open_connection for true async I/O.
    """

    __slots__ = (
        "_connect_timeout",
        "_max_retries",
        "_retry_delay",
        "_ssl_context",
        "_timeout",
    )

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._ssl_context = ssl_context or ssl.create_default_context()

    def _parse_url(self, url: str) -> tuple[str, int, str, bool]:
        """Parse URL into components."""
        parsed = urlparse(url)
        is_https = parsed.scheme == "https"
        host = parsed.hostname or ""
        port = parsed.port or (443 if is_https else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return host, port, path, is_https

    def _build_request(
        self,
        method: str,
        host: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> bytes:
        """Build HTTP/1.1 request bytes."""
        lines = [f"{method} {path} HTTP/1.1"]
        headers["Host"] = host
        if "Accept-Encoding" not in headers:
            headers["Accept-Encoding"] = "gzip, deflate"
        if "Connection" not in headers:
            headers["Connection"] = "close"
        if body and "Content-Length" not in headers:
            headers["Content-Length"] = str(len(body))

        for key, value in headers.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("")

        request = "\r\n".join(lines).encode("utf-8")
        if body:
            request += body
        return request

    async def _read_response_headers(
        self, reader: asyncio.StreamReader
    ) -> tuple[int, dict[str, str]]:
        """Read and parse HTTP response headers."""
        # Read status line
        status_line_bytes = await reader.readline()
        if not status_line_bytes:
            raise ConnectionError("Empty response from server")

        status_line = status_line_bytes.decode("utf-8", errors="replace").strip()
        parts = status_line.split(" ", 2)
        if len(parts) < 2:
            raise ConnectionError(f"Invalid status line: {status_line}")
        status_code = int(parts[1])

        # Read headers
        headers: dict[str, str] = {}
        while True:
            line_bytes = await reader.readline()
            if not line_bytes or line_bytes == b"\r\n" or line_bytes == b"\n":
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        return status_code, headers

    def _decompress(self, data: bytes, encoding: str | None) -> bytes:
        """Decompress response body if needed."""
        if not encoding:
            return data
        encoding = encoding.lower()
        if encoding == "gzip":
            return gzip.decompress(data)
        elif encoding == "deflate":
            try:
                return zlib.decompress(data)
            except zlib.error:
                return zlib.decompress(data, -zlib.MAX_WBITS)
        return data

    async def _read_body(
        self,
        reader: asyncio.StreamReader,
        headers: dict[str, str],
    ) -> bytes:
        """Read response body based on headers."""
        # Check for chunked transfer encoding
        transfer_encoding = headers.get("transfer-encoding", "").lower()
        if "chunked" in transfer_encoding:
            return await self._read_chunked_body(reader)

        # Check for content-length
        content_length = headers.get("content-length")
        if content_length:
            return await reader.readexactly(int(content_length))

        # Read until EOF
        return await reader.read()

    async def _read_chunked_body(self, reader: asyncio.StreamReader) -> bytes:
        """Read chunked transfer encoding body."""
        body = bytearray()
        while True:
            # Read chunk size line
            size_line = await reader.readline()
            size_str = size_line.decode("utf-8").strip()
            if not size_str:
                continue
            # Handle chunk extensions (after semicolon)
            if ";" in size_str:
                size_str = size_str.split(";")[0]
            chunk_size = int(size_str, 16)

            if chunk_size == 0:
                # Read trailing CRLF
                await reader.readline()
                break

            # Read chunk data
            chunk = await reader.readexactly(chunk_size)
            body.extend(chunk)
            # Read trailing CRLF
            await reader.readline()

        return bytes(body)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> AsyncHTTPResponse | AsyncIterator[bytes]:
        """
        Make an async HTTP request.

        Args:
            method: HTTP method
            url: Full URL
            headers: Request headers
            body: Request body as bytes
            timeout: Request timeout
            stream: If True, return async iterator for streaming

        Returns:
            AsyncHTTPResponse for non-streaming, AsyncIterator[bytes] for streaming
        """
        host, port, path, is_https = self._parse_url(url)
        request_headers = dict(headers) if headers else {}
        timeout = timeout or self._timeout

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            reader: asyncio.StreamReader | None = None
            writer: asyncio.StreamWriter | None = None
            try:
                # Connect with timeout
                ssl_ctx = self._ssl_context if is_https else None
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port, ssl=ssl_ctx),
                    timeout=self._connect_timeout,
                )
                # Assertions for type narrowing - these are guaranteed after successful open_connection
                assert reader is not None
                assert writer is not None

                # Send request
                request_bytes = self._build_request(method, host, path, request_headers, body)
                writer.write(request_bytes)
                await writer.drain()

                # Read response headers
                status_code, resp_headers = await asyncio.wait_for(
                    self._read_response_headers(reader),
                    timeout=timeout,
                )

                request_id = resp_headers.get("x-request-id") or resp_headers.get("request-id")

                if stream:
                    # Return async streaming iterator
                    return self._stream_response_async(reader, writer, resp_headers)

                # Read full body
                response_body = await asyncio.wait_for(
                    self._read_body(reader, resp_headers),
                    timeout=timeout,
                )

                # Decompress if needed
                content_encoding = resp_headers.get("content-encoding")
                response_body = self._decompress(response_body, content_encoding)

                # Close connection
                writer.close()
                await writer.wait_closed()

                return AsyncHTTPResponse(
                    status_code=status_code,
                    headers=resp_headers,
                    body=response_body,
                    request_id=request_id,
                )

            except builtins.TimeoutError:
                last_error = TimeoutError(
                    f"Request timed out after {timeout}s",
                    timeout_type="read",
                    timeout_seconds=timeout,
                )
                if writer:
                    writer.close()
            except OSError as e:
                last_error = ConnectionError(f"Connection failed: {e}")
                if writer:
                    writer.close()
            except Exception as e:
                last_error = ProviderAPIError(f"Request failed: {e}")
                if writer:
                    writer.close()

            # Exponential backoff with jitter
            if attempt < self._max_retries - 1:
                delay = self._retry_delay * (2**attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)

        raise last_error or ConnectionError("Request failed after retries")

    async def _stream_response_async(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        """Stream response body as async iterator."""
        try:
            transfer_encoding = headers.get("transfer-encoding", "").lower()
            is_chunked = "chunked" in transfer_encoding
            content_encoding = headers.get("content-encoding", "").lower()

            # Setup decompressor if needed
            decompressor = None
            if content_encoding == "gzip":
                decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            elif content_encoding == "deflate":
                decompressor = zlib.decompressobj(-zlib.MAX_WBITS)

            if is_chunked:
                async for chunk in self._stream_chunked(reader):
                    if decompressor:
                        chunk = decompressor.decompress(chunk)
                    yield chunk
                if decompressor:
                    remaining = decompressor.flush()
                    if remaining:
                        yield remaining
            else:
                while True:
                    chunk = await reader.read(8192)
                    if not chunk:
                        break
                    if decompressor:
                        chunk = decompressor.decompress(chunk)
                    yield chunk
                if decompressor:
                    remaining = decompressor.flush()
                    if remaining:
                        yield remaining
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _stream_chunked(self, reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
        """Stream chunked transfer encoding."""
        while True:
            size_line = await reader.readline()
            size_str = size_line.decode("utf-8").strip()
            if not size_str:
                continue
            if ";" in size_str:
                size_str = size_str.split(";")[0]
            chunk_size = int(size_str, 16)

            if chunk_size == 0:
                await reader.readline()
                break

            chunk = await reader.readexactly(chunk_size)
            yield chunk
            await reader.readline()

    async def post(
        self,
        url: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> AsyncHTTPResponse | AsyncIterator[bytes]:
        """Make an async POST request with JSON body."""
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        body = None
        if json_data is not None:
            body = json.dumps(json_data, ensure_ascii=False).encode("utf-8")

        return await self.request(
            "POST",
            url,
            headers=request_headers,
            body=body,
            timeout=timeout,
            stream=stream,
        )

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncHTTPResponse:
        """Make an async GET request."""
        result = await self.request("GET", url, headers=headers, timeout=timeout, stream=False)
        assert isinstance(result, AsyncHTTPResponse)
        return result

    async def close(self) -> None:
        """Close the client (no-op for now, connections close after use)."""
        pass

    async def __aenter__(self) -> AsyncHTTPClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
