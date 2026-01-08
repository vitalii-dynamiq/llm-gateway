"""
Tests for arcllm.http.sse module.
"""

from arcllm.http.sse import SSEEvent, SSEParser, parse_sse_stream


class TestSSEEvent:
    """Tests for SSEEvent dataclass."""

    def test_create_event(self):
        """Test basic event creation."""
        event = SSEEvent(data="hello", event="message")
        assert event.data == "hello"
        assert event.event == "message"

    def test_is_done(self):
        """Test is_done property."""
        done_event = SSEEvent(data="[DONE]")
        assert done_event.is_done is True

        regular_event = SSEEvent(data='{"content": "hello"}')
        assert regular_event.is_done is False

    def test_is_done_with_whitespace(self):
        """Test is_done handles whitespace."""
        event = SSEEvent(data="  [DONE]  ")
        assert event.is_done is True


class TestSSEParser:
    """Tests for SSEParser."""

    def test_parse_simple_event(self):
        """Test parsing a simple SSE event."""
        parser = SSEParser()
        events = list(parser.feed("data: hello\n\n"))
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_parse_event_with_type(self):
        """Test parsing event with type."""
        parser = SSEParser()
        events = list(parser.feed("event: message\ndata: hello\n\n"))
        assert len(events) == 1
        assert events[0].event == "message"
        assert events[0].data == "hello"

    def test_parse_event_with_id(self):
        """Test parsing event with id."""
        parser = SSEParser()
        events = list(parser.feed("id: 123\ndata: hello\n\n"))
        assert len(events) == 1
        assert events[0].id == "123"
        assert events[0].data == "hello"

    def test_parse_multiline_data(self):
        """Test parsing multi-line data."""
        parser = SSEParser()
        events = list(parser.feed("data: line1\ndata: line2\n\n"))
        assert len(events) == 1
        assert events[0].data == "line1\nline2"

    def test_parse_multiple_events(self):
        """Test parsing multiple events."""
        parser = SSEParser()
        events = list(parser.feed("data: first\n\ndata: second\n\n"))
        assert len(events) == 2
        assert events[0].data == "first"
        assert events[1].data == "second"

    def test_parse_json_data(self):
        """Test parsing JSON data."""
        parser = SSEParser()
        json_str = '{"content": "hello"}'
        events = list(parser.feed(f"data: {json_str}\n\n"))
        assert len(events) == 1
        assert events[0].data == json_str

    def test_skip_comments(self):
        """Test comments are skipped."""
        parser = SSEParser()
        events = list(parser.feed(": this is a comment\ndata: hello\n\n"))
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_handle_bytes_input(self):
        """Test handling bytes input."""
        parser = SSEParser()
        events = list(parser.feed(b"data: hello\n\n"))
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_handle_partial_chunks(self):
        """Test handling partial chunks."""
        parser = SSEParser()

        # First chunk - incomplete
        events = list(parser.feed("data: hel"))
        assert len(events) == 0

        # Second chunk - complete the event
        events = list(parser.feed("lo\n\n"))
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_handle_unicode(self):
        """Test handling unicode data."""
        parser = SSEParser()
        events = list(parser.feed("data: Hello 世界 🌍\n\n"))
        assert len(events) == 1
        assert events[0].data == "Hello 世界 🌍"

    def test_handle_crlf(self):
        """Test handling CRLF line endings."""
        parser = SSEParser()
        events = list(parser.feed("data: hello\r\n\r\n"))
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_data_with_colon(self):
        """Test data containing colons."""
        parser = SSEParser()
        events = list(parser.feed("data: time: 12:30:00\n\n"))
        assert len(events) == 1
        assert events[0].data == "time: 12:30:00"

    def test_flush(self):
        """Test flushing remaining data."""
        parser = SSEParser()
        # Feed incomplete event
        list(parser.feed("data: hello"))
        # Flush
        final = parser.flush()
        assert final is not None
        assert final.data == "hello"

    def test_flush_empty(self):
        """Test flushing when empty."""
        parser = SSEParser()
        final = parser.flush()
        assert final is None


class TestParseSSEStream:
    """Tests for parse_sse_stream helper function."""

    def test_parse_stream(self):
        """Test parsing byte stream."""

        def stream():
            yield b"data: first\n\n"
            yield b"data: second\n\n"

        events = list(parse_sse_stream(stream()))
        assert len(events) == 2
        assert events[0].data == "first"
        assert events[1].data == "second"

    def test_parse_chunked_stream(self):
        """Test parsing chunked byte stream."""

        def stream():
            yield b"data: hel"
            yield b"lo\n\n"
            yield b"data: world\n"
            yield b"\n"

        events = list(parse_sse_stream(stream()))
        assert len(events) == 2
        assert events[0].data == "hello"
        assert events[1].data == "world"

    def test_parse_openai_style_stream(self):
        """Test parsing OpenAI-style SSE stream."""

        def stream():
            yield b'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}\n\n'
            yield b'data: {"id":"1","choices":[{"delta":{"content":"!"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        events = list(parse_sse_stream(stream()))
        assert len(events) == 3
        assert events[2].is_done is True
