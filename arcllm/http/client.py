"""
Synchronous HTTP client using stdlib http.client.

Features:
- Connection pooling (per-host)
- TLS support
- Timeout handling
- Streaming response support
- Proxy support via HTTP(S)_PROXY env vars
- Gzip/deflate decompression
- Retries with exponential backoff and jitter
"""

from __future__ import annotations

import builtins
import contextlib
import gzip
import http.client
import json
import os
import random
import ssl
import time
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from arcllm.exceptions import (
    ConnectionError,
    ProviderAPIError,
    TimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["HTTPClient", "HTTPResponse"]


@dataclass(slots=True)
class HTTPResponse:
    """Response from an HTTP request."""

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


@dataclass
class ConnectionPool:
    """Simple connection pool for a single host."""

    host: str
    port: int
    is_https: bool
    connections: list[http.client.HTTPConnection | http.client.HTTPSConnection] = field(
        default_factory=lambda: []
    )
    max_connections: int = 10

    def get_connection(
        self,
        timeout: float,
        ssl_context: ssl.SSLContext | None = None,
    ) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        """Get a connection from the pool or create a new one."""
        # Try to reuse existing connection
        while self.connections:
            conn = self.connections.pop()
            try:
                # Test if connection is still alive
                conn.sock  # noqa: B018 - just accessing to check
                return conn
            except Exception:
                # Connection is dead, try next
                continue

        # Create new connection
        if self.is_https:
            ctx = ssl_context or ssl.create_default_context()
            conn = http.client.HTTPSConnection(
                self.host,
                self.port,
                timeout=timeout,
                context=ctx,
            )
        else:
            conn = http.client.HTTPConnection(
                self.host,
                self.port,
                timeout=timeout,
            )
        return conn

    def return_connection(
        self, conn: http.client.HTTPConnection | http.client.HTTPSConnection
    ) -> None:
        """Return a connection to the pool."""
        if len(self.connections) < self.max_connections:
            self.connections.append(conn)
        else:
            conn.close()

    def close_all(self) -> None:
        """Close all connections in the pool."""
        for conn in self.connections:
            with contextlib.suppress(Exception):
                conn.close()
        self.connections.clear()


class HTTPClient:
    """
    Synchronous HTTP client with connection pooling and retry support.

    Thread-safe for use from multiple threads (each thread gets own connection).
    """

    __slots__ = (
        "_connect_timeout",
        "_max_retries",
        "_pools",
        "_proxy_url",
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
        self._pools: dict[str, ConnectionPool] = {}
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._ssl_context = ssl_context or ssl.create_default_context()
        self._proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    def _get_pool(self, url: str) -> tuple[ConnectionPool, str]:
        """Get or create connection pool for URL, return pool and path."""
        parsed = urlparse(url)
        is_https = parsed.scheme == "https"
        host = parsed.hostname or ""
        port = parsed.port or (443 if is_https else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        pool_key = f"{parsed.scheme}://{host}:{port}"
        if pool_key not in self._pools:
            self._pools[pool_key] = ConnectionPool(
                host=host,
                port=port,
                is_https=is_https,
            )
        return self._pools[pool_key], path

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
                # Some servers send raw deflate without zlib header
                return zlib.decompress(data, -zlib.MAX_WBITS)
        return data

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> HTTPResponse | Iterator[bytes]:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
            headers: Request headers
            body: Request body as bytes
            timeout: Request timeout (overrides default)
            stream: If True, return iterator for streaming response

        Returns:
            HTTPResponse for non-streaming, Iterator[bytes] for streaming
        """
        pool, path = self._get_pool(url)
        request_headers = {
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        if headers:
            request_headers.update(headers)

        timeout = timeout or self._timeout
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            conn: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
            try:
                conn = pool.get_connection(timeout, self._ssl_context)
                conn.request(method, path, body=body, headers=request_headers)
                response = conn.getresponse()

                # Extract request ID from headers
                request_id = response.getheader("x-request-id") or response.getheader("request-id")

                if stream:
                    # Return streaming iterator
                    return self._stream_response(conn, response, pool)

                # Read full response
                response_body = response.read()
                content_encoding = response.getheader("Content-Encoding")
                response_body = self._decompress(response_body, content_encoding)

                # Return connection to pool
                pool.return_connection(conn)
                conn = None

                # Build headers dict
                resp_headers = {k.lower(): v for k, v in response.getheaders()}

                return HTTPResponse(
                    status_code=response.status,
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
                if conn:
                    with contextlib.suppress(Exception):
                        conn.close()
            except OSError as e:
                last_error = ConnectionError(f"Connection failed: {e}")
                if conn:
                    with contextlib.suppress(Exception):
                        conn.close()
            except Exception as e:
                last_error = ProviderAPIError(f"Request failed: {e}")
                if conn:
                    with contextlib.suppress(Exception):
                        conn.close()

            # Exponential backoff with jitter
            if attempt < self._max_retries - 1:
                delay = self._retry_delay * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

        raise last_error or ConnectionError("Request failed after retries")

    def _stream_response(
        self,
        conn: http.client.HTTPConnection | http.client.HTTPSConnection,
        response: http.client.HTTPResponse,
        pool: ConnectionPool,
    ) -> Iterator[bytes]:
        """Stream response body as iterator of bytes."""
        try:
            content_encoding = response.getheader("Content-Encoding")
            if content_encoding and content_encoding.lower() == "gzip":
                # For gzip, we need to decompress incrementally
                decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        # Flush remaining data
                        remaining = decompressor.flush()
                        if remaining:
                            yield remaining
                        break
                    yield decompressor.decompress(chunk)
            else:
                # Read in chunks
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    yield chunk
        finally:
            # Don't return streaming connections to pool
            with contextlib.suppress(Exception):
                conn.close()

    def post(
        self,
        url: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> HTTPResponse | Iterator[bytes]:
        """Make a POST request with JSON body."""
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        body = None
        if json_data is not None:
            body = json.dumps(json_data, ensure_ascii=False).encode("utf-8")

        return self.request(
            "POST",
            url,
            headers=request_headers,
            body=body,
            timeout=timeout,
            stream=stream,
        )

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse:
        """Make a GET request."""
        result = self.request("GET", url, headers=headers, timeout=timeout, stream=False)
        assert isinstance(result, HTTPResponse)
        return result

    def close(self) -> None:
        """Close all connections."""
        for pool in self._pools.values():
            pool.close_all()
        self._pools.clear()

    def __enter__(self) -> HTTPClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
