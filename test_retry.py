"""Tests for API retry logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from providers import _call_with_retry, _is_retryable, _get_retry_after


# --- Mock exceptions ---

class MockRateLimitError(Exception):
    pass

# Give it the class name the retry logic checks for
MockRateLimitError.__name__ = "RateLimitError"


class MockInternalServerError(Exception):
    pass

MockInternalServerError.__name__ = "InternalServerError"


class MockAPIConnectionError(Exception):
    pass

MockAPIConnectionError.__name__ = "APIConnectionError"


class MockBadRequestError(Exception):
    """Non-retryable error."""
    pass

MockBadRequestError.__name__ = "BadRequestError"


class MockHTTPError(Exception):
    """Error with a status_code attribute."""
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class MockErrorWithRetryAfter(Exception):
    """Error with a Retry-After header."""
    def __init__(self, retry_after):
        self.response = MagicMock()
        self.response.headers = {"retry-after": str(retry_after)}
        super().__init__("rate limited")

MockErrorWithRetryAfter.__name__ = "RateLimitError"


class TestIsRetryable:
    def test_rate_limit_error(self):
        assert _is_retryable(MockRateLimitError()) is True

    def test_internal_server_error(self):
        assert _is_retryable(MockInternalServerError()) is True

    def test_connection_error(self):
        assert _is_retryable(MockAPIConnectionError()) is True

    def test_bad_request_not_retryable(self):
        assert _is_retryable(MockBadRequestError()) is False

    def test_status_code_429(self):
        assert _is_retryable(MockHTTPError(429)) is True

    def test_status_code_500(self):
        assert _is_retryable(MockHTTPError(500)) is True

    def test_status_code_502(self):
        assert _is_retryable(MockHTTPError(502)) is True

    def test_status_code_503(self):
        assert _is_retryable(MockHTTPError(503)) is True

    def test_status_code_529(self):
        assert _is_retryable(MockHTTPError(529)) is True

    def test_status_code_400_not_retryable(self):
        assert _is_retryable(MockHTTPError(400)) is False

    def test_status_code_401_not_retryable(self):
        assert _is_retryable(MockHTTPError(401)) is False

    def test_python_connection_error(self):
        assert _is_retryable(ConnectionError()) is True

    def test_python_timeout_error(self):
        assert _is_retryable(TimeoutError()) is True

    def test_value_error_not_retryable(self):
        assert _is_retryable(ValueError("bad")) is False


class TestGetRetryAfter:
    def test_extracts_retry_after(self):
        exc = MockErrorWithRetryAfter(5.0)
        assert _get_retry_after(exc) == 5.0

    def test_no_response_attribute(self):
        assert _get_retry_after(ValueError("no response")) is None

    def test_no_headers(self):
        exc = MagicMock()
        exc.response.headers = {}
        assert _get_retry_after(exc) is None


class TestCallWithRetry:
    @patch("providers.time.sleep")
    def test_succeeds_first_try(self, mock_sleep):
        fn = MagicMock(return_value="result")
        assert _call_with_retry(fn, "arg1", key="val") == "result"
        fn.assert_called_once_with("arg1", key="val")
        mock_sleep.assert_not_called()

    @patch("providers.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep):
        fn = MagicMock(side_effect=[MockRateLimitError(), "result"])
        assert _call_with_retry(fn, max_retries=3) == "result"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("providers.time.sleep")
    def test_retries_on_server_error(self, mock_sleep):
        fn = MagicMock(side_effect=[MockInternalServerError(), "result"])
        assert _call_with_retry(fn, max_retries=3) == "result"
        assert fn.call_count == 2

    @patch("providers.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        fn = MagicMock(side_effect=[ConnectionError(), "result"])
        assert _call_with_retry(fn, max_retries=3) == "result"
        assert fn.call_count == 2

    @patch("providers.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        fn = MagicMock(side_effect=MockRateLimitError())
        with pytest.raises(MockRateLimitError):
            _call_with_retry(fn, max_retries=2)
        assert fn.call_count == 3  # initial + 2 retries

    @patch("providers.time.sleep")
    def test_no_retry_on_bad_request(self, mock_sleep):
        fn = MagicMock(side_effect=MockBadRequestError())
        with pytest.raises(MockBadRequestError):
            _call_with_retry(fn, max_retries=3)
        assert fn.call_count == 1  # no retries
        mock_sleep.assert_not_called()

    @patch("providers.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        fn = MagicMock(side_effect=[
            MockRateLimitError(), MockRateLimitError(), MockRateLimitError(), "result",
        ])
        _call_with_retry(fn, max_retries=3)

        # Delays should be 1, 2, 4 (2^0, 2^1, 2^2)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1, 2, 4]

    @patch("providers.time.sleep")
    def test_respects_retry_after_header(self, mock_sleep):
        fn = MagicMock(side_effect=[MockErrorWithRetryAfter(10.0), "result"])
        _call_with_retry(fn, max_retries=3)

        # Should use the Retry-After value instead of exponential backoff
        mock_sleep.assert_called_once_with(10.0)

    @patch("providers.time.sleep")
    def test_backoff_caps_at_30s(self, mock_sleep):
        """Exponential backoff should cap at 30 seconds."""
        errors = [MockRateLimitError()] * 6 + ["result"]
        fn = MagicMock(side_effect=errors)
        _call_with_retry(fn, max_retries=6)

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32->30 (capped)
        assert delays == [1, 2, 4, 8, 16, 30]

    @patch("providers.time.sleep")
    def test_passes_through_args_and_kwargs(self, mock_sleep):
        fn = MagicMock(return_value="ok")
        _call_with_retry(fn, "a", "b", max_retries=3, x=1, y=2)
        fn.assert_called_once_with("a", "b", x=1, y=2)
