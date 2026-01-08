"""
Tests for fastlitellm.http.async_client module.

Tests the asynchronous HTTP client with mocked network calls.
"""

from __future__ import annotations

import pytest

from fastlitellm.http.async_client import AsyncHTTPClient, AsyncHTTPResponse


class TestAsyncHTTPResponse:
    """Tests for AsyncHTTPResponse dataclass."""

    def test_create_response(self):
        """Test creating a basic async response."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"hello": "world"}',
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_json_method(self):
        """Test JSON parsing."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=b'{"hello": "world"}',
        )
        data = response.json()
        assert data == {"hello": "world"}

    def test_text_property(self):
        """Test text decoding."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=b"Hello, World!",
        )
        assert response.text == "Hello, World!"

    def test_request_id(self):
        """Test request ID extraction."""
        response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=b"",
            request_id="req-123",
        )
        assert response.request_id == "req-123"


class TestAsyncHTTPClient:
    """Tests for AsyncHTTPClient."""

    def test_create_client(self):
        """Test creating an async HTTP client."""
        client = AsyncHTTPClient(
            timeout=30.0,
            connect_timeout=5.0,
            max_retries=3,
        )
        assert client._timeout == 30.0
        assert client._connect_timeout == 5.0
        assert client._max_retries == 3

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the client."""
        client = AsyncHTTPClient()
        await client.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as async context manager."""
        async with AsyncHTTPClient() as client:
            assert client is not None

    def test_decompress_gzip(self):
        """Test gzip decompression."""
        import gzip

        client = AsyncHTTPClient()
        original = b"Hello, World!"
        compressed = gzip.compress(original)
        result = client._decompress(compressed, "gzip")
        assert result == original

    def test_decompress_deflate(self):
        """Test deflate decompression."""
        import zlib

        client = AsyncHTTPClient()
        original = b"Hello, World!"
        compressed = zlib.compress(original)
        result = client._decompress(compressed, "deflate")
        assert result == original

    def test_decompress_none(self):
        """Test no decompression when encoding is None."""
        client = AsyncHTTPClient()
        original = b"Hello, World!"
        result = client._decompress(original, None)
        assert result == original


class TestAsyncHTTPClientParsing:
    """Tests for AsyncHTTPClient URL parsing."""

    @pytest.fixture
    def client(self):
        """Create an async HTTP client."""
        return AsyncHTTPClient()

    def test_parse_https_url(self, client):
        """Test parsing HTTPS URL."""
        host, port, path, is_https = client._parse_url(
            "https://api.example.com/v1/chat/completions"
        )
        assert host == "api.example.com"
        assert port == 443
        assert path == "/v1/chat/completions"
        assert is_https is True

    def test_parse_http_url(self, client):
        """Test parsing HTTP URL."""
        host, port, path, is_https = client._parse_url("http://localhost:8080/api")
        assert host == "localhost"
        assert port == 8080
        assert path == "/api"
        assert is_https is False

    def test_parse_url_with_query(self, client):
        """Test parsing URL with query parameters."""
        _host, _port, path, _is_https = client._parse_url(
            "https://api.example.com/v1?key=value&other=param"
        )
        assert path == "/v1?key=value&other=param"

    def test_parse_url_default_port_https(self, client):
        """Test default port for HTTPS."""
        _host, port, _path, _is_https = client._parse_url("https://api.example.com/test")
        assert port == 443

    def test_parse_url_default_port_http(self, client):
        """Test default port for HTTP."""
        _host, port, _path, _is_https = client._parse_url("http://api.example.com/test")
        assert port == 80

    def test_parse_url_empty_path(self, client):
        """Test URL with empty path."""
        _host, _port, path, _is_https = client._parse_url("https://api.example.com")
        assert path == "/"
