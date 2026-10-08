"""Tests for retry functionality."""
# pylint: disable=W0621,W0212,W0613

import logging
from typing import Any
from unittest.mock import Mock, patch

import pytest
from httpx import Response
from openai import OpenAIError, RateLimitError

from igsr.agent.retry import RetryConfig, RetryHandler


def create_mock_structlog_logger() -> Mock:
    """Create a properly configured mock structlog logger.

    Returns:
        Mock: Configured mock logger with standard logging methods.
    """
    mock_logger = Mock()
    # Add standard logging methods that structlog provides:
    for method in ["debug", "info", "warning", "error", "critical"]:
        setattr(mock_logger, method, Mock())
    return mock_logger


def create_rate_limit_error() -> RateLimitError:
    """Create a properly configured RateLimitError for testing.

    Returns:
        RateLimitError: Configured error instance with proper response and body.
    """
    request = Mock()
    request.method = "POST"
    request.url = "https://api.openai.com/v1/chat/completions"

    response = Response(
        status_code=429,
        headers={"retry-after": "30"},
        json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
        request=request,
    )

    return RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
    )


def create_mock_function(name: str = "test_func") -> Mock:
    """Create a properly configured mock function for testing.

    Args:
        name (str, optional): Name for the function. Defaults to "test_func".

    Returns:
        Mock: Configured mock function with proper name attributes.
    """
    mock_func = Mock()
    mock_func.__name__ = name
    mock_func.__qualname__ = f"TestModule.{name}"
    return mock_func


class TestRetryConfig:
    """Tests for RetryConfig functionality."""

    def test_default_values(self) -> None:
        """Test that default configuration values are set correctly."""
        config = RetryConfig()
        assert config.max_retries == 5
        assert config.min_wait == 1
        assert config.max_wait == 60
        assert config.logger is None

    def test_custom_values(self) -> None:
        """Test that custom configuration values are properly set."""
        logger = create_mock_structlog_logger()
        config = RetryConfig(max_retries=3, min_wait=2, max_wait=30, logger=logger)
        assert config.max_retries == 3
        assert config.min_wait == 2
        assert config.max_wait == 30
        assert config.logger is logger


class TestRetryHandler:
    """Tests for RetryHandler functionality."""

    @pytest.fixture
    def mock_sleep(self) -> Mock:  # type: ignore
        """Fixture to mock time.sleep.

        Returns:
            Mock: The mock sleep function.
        """
        with patch("time.sleep") as mock:
            yield mock

    @pytest.fixture
    def handler(self) -> RetryHandler:
        """Fixture providing a RetryHandler with default config.

        Returns:
            RetryHandler: Configured handler instance.
        """
        return RetryHandler(RetryConfig())

    def test_successful_execution(self, handler: RetryHandler, mock_sleep: Mock) -> None:
        """Test execution of function that succeeds without retries."""
        mock_func = create_mock_function("successful_func")
        mock_func.return_value = "success"
        decorated = handler.with_retries()(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1
        assert not mock_sleep.called

    def test_retry_on_rate_limit(self, mock_sleep: Mock) -> None:
        """Test retrying on rate limit errors with structured logging."""
        logger = create_mock_structlog_logger()
        handler = RetryHandler(RetryConfig(logger=logger))

        # Mock function that fails twice then succeeds:
        error = create_rate_limit_error()
        mock_func = create_mock_function("retry_test")
        mock_func.side_effect = [error, error, "success"]
        decorated = handler.with_retries()(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3  # Two failures + success.
        assert mock_sleep.call_count == 2  # Sleep between attempts.

        # Verify logging:
        logger.warning.assert_called_with(
            "retrying_api_call",
            attempt=2,  # Last retry attempt.
            next_attempt_in=mock_sleep.call_args[0][0],
            error="Rate limit exceeded",
        )

    def test_max_retries_exceeded(self, mock_sleep: Mock) -> None:
        """Test that retries stop after max_retries attempts."""
        handler = RetryHandler(RetryConfig(max_retries=3))

        # Create properly configured mock function that always fails:
        error = create_rate_limit_error()
        mock_func = create_mock_function("always_fails")
        mock_func.side_effect = error
        decorated = handler.with_retries()(mock_func)

        with pytest.raises(RateLimitError):
            decorated()

        assert mock_func.call_count == 3  # Initial attempt + 2 retries.
        assert mock_sleep.call_count == 2  # Sleep between attempts.

    def test_exponential_backoff(self, mock_sleep: Mock) -> None:
        """Test that backoff times increase exponentially."""
        handler = RetryHandler(RetryConfig(min_wait=1, max_wait=10))

        # Function that fails three times then succeeds:
        error = create_rate_limit_error()
        mock_func = create_mock_function("exponential_test")
        mock_func.side_effect = [error, error, error, "success"]
        decorated = handler.with_retries()(mock_func)

        decorated()

        # Check exponential increase in sleep times:
        sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
        assert len(sleep_times) == 3
        assert sleep_times[0] < sleep_times[1] < sleep_times[2]

    def test_max_wait_respected(self, mock_sleep: Mock) -> None:
        """Test that wait time never exceeds max_wait."""
        max_wait = 5
        handler = RetryHandler(RetryConfig(min_wait=1, max_wait=max_wait))

        # Function that fails four times then succeeds:
        error = create_rate_limit_error()
        mock_func = create_mock_function("max_wait_test")
        mock_func.side_effect = [error] * 4 + ["success"]
        decorated = handler.with_retries()(mock_func)

        decorated()

        sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
        assert all(t <= max_wait for t in sleep_times)

    def test_retry_conditions(self, handler: RetryHandler) -> None:
        """Test which exceptions trigger retries."""
        # Rate limit errors should retry:
        error = create_rate_limit_error()
        assert handler._should_retry(error)

        # Server errors should retry:
        error_500 = OpenAIError()
        error_500.status_code = 500  # type: ignore
        assert handler._should_retry(error_500)

        # Client timeout should retry:
        error_408 = OpenAIError()
        error_408.status_code = 408  # type: ignore
        assert handler._should_retry(error_408)

        # Other client errors should not retry:
        error_400 = OpenAIError()
        error_400.status_code = 400  # type: ignore
        assert not handler._should_retry(error_400)

        # Non-OpenAI errors should not retry:
        assert not handler._should_retry(ValueError("Random error"))

    def test_no_logger_fallback(self, mock_sleep: Mock, caplog: Any) -> None:
        """Test fallback to standard logging when no logger provided."""
        handler = RetryHandler(RetryConfig(logger=None))

        # Function that fails once then succeeds:
        error = create_rate_limit_error()
        mock_func = create_mock_function("fallback_test")
        mock_func.side_effect = [error, "success"]
        decorated = handler.with_retries()(mock_func)

        with caplog.at_level(logging.WARNING):
            decorated()

        # Verify standard logging was used:
        assert any(
            record.levelname == "WARNING" and "Retrying TestModule.fallback_test" in record.message
            for record in caplog.records
        )
