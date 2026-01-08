"""
Tests for fastlitellm.http.client module.

Tests the synchronous HTTP client with mocked network calls.
"""

from __future__ import annotations

import http.client
from unittest.mock import MagicMock

import pytest

from fastlitellm.http.client import ConnectionPool, HTTPClient, HTTPResponse


class TestHTTPResponse:
    """Tests for HTTPResponse dataclass."""

    def test_create_response(self):
        """Test creating a basic response."""
        response = HTTPResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"hello": "world"}',
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_json_method(self):
        """Test JSON parsing."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b'{"hello": "world"}',
        )
        data = response.json()
        assert data == {"hello": "world"}

    def test_text_property(self):
        """Test text decoding."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b"Hello, World!",
        )
        assert response.text == "Hello, World!"

    def test_request_id(self):
        """Test request ID extraction."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b"",
            request_id="req-123",
        )
        assert response.request_id == "req-123"


class TestConnectionPool:
    """Tests for ConnectionPool."""

    def test_create_pool(self):
        """Test creating a connection pool."""
        pool = ConnectionPool(
            host="api.example.com",
            port=443,
            is_https=True,
        )
        assert pool.host == "api.example.com"
        assert pool.port == 443
        assert pool.is_https is True
        assert len(pool.connections) == 0

    def test_get_connection_creates_new(self):
        """Test getting a connection when pool is empty."""
        pool = ConnectionPool(
            host="api.example.com",
            port=443,
            is_https=True,
        )
        conn = pool.get_connection(timeout=30.0)
        assert conn is not None

    def test_return_connection(self):
        """Test returning a connection to the pool."""
        pool = ConnectionPool(
            host="api.example.com",
            port=443,
            is_https=True,
            max_connections=5,
        )
        mock_conn = MagicMock(spec=http.client.HTTPSConnection)
        pool.return_connection(mock_conn)
        assert len(pool.connections) == 1

    def test_max_connections_respected(self):
        """Test that max_connections limit is respected."""
        pool = ConnectionPool(
            host="api.example.com",
            port=443,
            is_https=True,
            max_connections=2,
        )
        # Add more connections than max
        for _ in range(5):
            mock_conn = MagicMock(spec=http.client.HTTPSConnection)
            pool.return_connection(mock_conn)

        assert len(pool.connections) == 2

    def test_close_all(self):
        """Test closing all connections."""
        pool = ConnectionPool(
            host="api.example.com",
            port=443,
            is_https=True,
        )
        mock_conn1 = MagicMock(spec=http.client.HTTPSConnection)
        mock_conn2 = MagicMock(spec=http.client.HTTPSConnection)
        pool.return_connection(mock_conn1)
        pool.return_connection(mock_conn2)

        pool.close_all()

        assert len(pool.connections) == 0
        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_called_once()


class TestHTTPClient:
    """Tests for HTTPClient."""

    def test_create_client(self):
        """Test creating an HTTP client."""
        client = HTTPClient(
            timeout=30.0,
            connect_timeout=5.0,
            max_retries=3,
        )
        assert client._timeout == 30.0
        assert client._connect_timeout == 5.0
        assert client._max_retries == 3

    def test_close_client(self):
        """Test closing the client."""
        client = HTTPClient()
        # Add a mock pool
        client._pools["https://api.example.com:443"] = MagicMock()
        client.close()
        assert len(client._pools) == 0

    def test_context_manager(self):
        """Test using client as context manager."""
        with HTTPClient() as client:
            assert client is not None
        # After exit, pools should be cleared

    def test_decompress_gzip(self):
        """Test gzip decompression."""
        import gzip

        client = HTTPClient()
        original = b"Hello, World!"
        compressed = gzip.compress(original)
        result = client._decompress(compressed, "gzip")
        assert result == original

    def test_decompress_deflate(self):
        """Test deflate decompression."""
        import zlib

        client = HTTPClient()
        original = b"Hello, World!"
        compressed = zlib.compress(original)
        result = client._decompress(compressed, "deflate")
        assert result == original

    def test_decompress_none(self):
        """Test no decompression when encoding is None."""
        client = HTTPClient()
        original = b"Hello, World!"
        result = client._decompress(original, None)
        assert result == original


class TestHTTPClientIntegration:
    """Integration-style tests for HTTPClient with mocked connections."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a client with mocked connection."""
        client = HTTPClient()
        return client

    def test_get_pool(self, mock_http_client):
        """Test getting connection pool for URL."""
        pool, path = mock_http_client._get_pool("https://api.example.com/v1/chat/completions")
        assert pool.host == "api.example.com"
        assert pool.port == 443
        assert pool.is_https is True
        assert path == "/v1/chat/completions"

    def test_get_pool_with_query(self, mock_http_client):
        """Test getting pool with query parameters."""
        _pool, path = mock_http_client._get_pool("https://api.example.com/v1?key=value")
        assert path == "/v1?key=value"

    def test_get_pool_http(self, mock_http_client):
        """Test getting pool for HTTP URL."""
        pool, _path = mock_http_client._get_pool("http://localhost:8080/api")
        assert pool.host == "localhost"
        assert pool.port == 8080
        assert pool.is_https is False
