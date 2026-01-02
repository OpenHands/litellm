"""
Tests for JSON logging functionality in litellm/_logging.py

This tests the fix for asyncio exception handling with JSON logging.
The issue was that asyncio exceptions were logged without stacktraces
because exc_info was set to None in the original implementation.
"""
import asyncio
import json
import logging
import os
import sys
from io import StringIO
from unittest.mock import patch

import pytest


class TestJsonLogging:
    """Tests for JSON logging configuration."""

    def test_json_formatter_includes_stacktrace_on_exception(self):
        """Test that JsonFormatter includes stacktrace when exc_info is provided."""
        # Import after setting JSON_LOGS
        with patch.dict(os.environ, {"JSON_LOGS": "true"}):
            # Force reimport to pick up JSON_LOGS
            if "litellm._logging" in sys.modules:
                del sys.modules["litellm._logging"]
            from litellm._logging import JsonFormatter

            formatter = JsonFormatter()
            
            # Create a log record with exception info
            try:
                raise RuntimeError("Test exception")
            except RuntimeError:
                exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=exc_info,
            )
            
            output = formatter.format(record)
            data = json.loads(output)
            
            assert "stacktrace" in data
            assert "RuntimeError" in data["stacktrace"]
            assert "Test exception" in data["stacktrace"]

    def test_json_formatter_no_stacktrace_without_exception(self):
        """Test that JsonFormatter doesn't include stacktrace when no exception."""
        with patch.dict(os.environ, {"JSON_LOGS": "true"}):
            if "litellm._logging" in sys.modules:
                del sys.modules["litellm._logging"]
            from litellm._logging import JsonFormatter

            formatter = JsonFormatter()
            
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            
            output = formatter.format(record)
            data = json.loads(output)
            
            assert "stacktrace" not in data
            assert data["message"] == "Test message"
            assert data["level"] == "INFO"

    def test_async_json_exception_handler_includes_traceback(self):
        """Test that async_json_exception_handler includes traceback from exception."""
        with patch.dict(os.environ, {"JSON_LOGS": "true"}):
            if "litellm._logging" in sys.modules:
                del sys.modules["litellm._logging"]
            from litellm._logging import (
                _get_json_error_handler,
                _set_json_error_handler,
                JsonFormatter,
            )

            # Capture output
            captured = StringIO()
            handler = logging.StreamHandler(captured)
            handler.setFormatter(JsonFormatter())
            
            # Set our custom handler
            _set_json_error_handler(handler)
            
            # Get the error handler function
            error_handler = _get_json_error_handler()
            
            # Simulate asyncio exception context with traceback
            try:
                raise ValueError("Async task failed")
            except ValueError as e:
                context = {
                    "message": "Task exception was never retrieved",
                    "exception": e,
                }
            
            # Create a mock loop
            class MockLoop:
                def default_exception_handler(self, context):
                    pass
            
            # Get the actual exception handler function from _logging module
            from litellm._logging import _async_json_exception_handler
            _async_json_exception_handler(MockLoop(), context)
            
            output = captured.getvalue()
            data = json.loads(output.strip())
            
            assert "stacktrace" in data
            assert "ValueError" in data["stacktrace"]
            assert "Async task failed" in data["stacktrace"]

    def test_async_json_exception_handler_handles_non_exception_context(self):
        """Test that async_json_exception_handler handles context without exception."""
        with patch.dict(os.environ, {"JSON_LOGS": "true"}):
            if "litellm._logging" in sys.modules:
                del sys.modules["litellm._logging"]
            from litellm._logging import (
                _get_json_error_handler,
                _set_json_error_handler,
                JsonFormatter,
            )

            captured = StringIO()
            handler = logging.StreamHandler(captured)
            handler.setFormatter(JsonFormatter())
            
            # Set our custom handler
            _set_json_error_handler(handler)
            
            # Context without exception
            context = {
                "message": "Some asyncio warning",
            }
            
            class MockLoop:
                def default_exception_handler(self, context):
                    pass
            
            from litellm._logging import _async_json_exception_handler
            _async_json_exception_handler(MockLoop(), context)
            
            output = captured.getvalue()
            data = json.loads(output.strip())
            
            assert data["message"] == "Some asyncio warning"
            assert "stacktrace" not in data


@pytest.mark.asyncio
class TestAsyncioJsonExceptionHandler:
    """Tests for asyncio exception handler with JSON logging."""

    async def test_setup_asyncio_json_exception_handler_sets_handler(self):
        """Test that _setup_asyncio_json_exception_handler sets the handler on running loop."""
        with patch.dict(os.environ, {"JSON_LOGS": "true"}):
            if "litellm._logging" in sys.modules:
                del sys.modules["litellm._logging"]
            from litellm._logging import _setup_asyncio_json_exception_handler

            loop = asyncio.get_running_loop()
            original_handler = loop.get_exception_handler()
            
            try:
                _setup_asyncio_json_exception_handler()
                
                new_handler = loop.get_exception_handler()
                assert new_handler is not None
                assert new_handler != original_handler
            finally:
                # Restore original handler
                loop.set_exception_handler(original_handler)

    async def test_asyncio_exception_logged_as_json_with_stacktrace(self):
        """Test that asyncio exceptions are logged as JSON with stacktrace."""
        with patch.dict(os.environ, {"JSON_LOGS": "true"}):
            if "litellm._logging" in sys.modules:
                del sys.modules["litellm._logging"]
            from litellm._logging import (
                _setup_asyncio_json_exception_handler,
                _set_json_error_handler,
                JsonFormatter,
            )

            captured = StringIO()
            handler = logging.StreamHandler(captured)
            handler.setFormatter(JsonFormatter())
            
            # Set our custom handler to capture output
            _set_json_error_handler(handler)
            
            loop = asyncio.get_running_loop()
            original_handler = loop.get_exception_handler()
            
            try:
                _setup_asyncio_json_exception_handler()
                
                async def failing_task():
                    await asyncio.sleep(0.01)
                    raise RuntimeError("Test async exception")
                
                task = asyncio.create_task(failing_task())
                await asyncio.sleep(0.1)
                
                # Force garbage collection to trigger exception handler
                import gc
                del task
                gc.collect()
                await asyncio.sleep(0.1)
                
                output = captured.getvalue()
                if output:
                    data = json.loads(output.strip())
                    assert "stacktrace" in data
                    assert "RuntimeError" in data["stacktrace"]
                    assert "Test async exception" in data["stacktrace"]
            finally:
                loop.set_exception_handler(original_handler)
