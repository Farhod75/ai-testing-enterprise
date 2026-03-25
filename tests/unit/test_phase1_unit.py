"""
tests/unit/test_phase1_unit.py

Phase 1: Unit Testing of AI Components

What we test here:
1. ClaudeConfig — validation logic
2. ChatMessage — input validation
3. ClaudeResponse — computed properties
4. Validators — every validator function
5. ClaudeClient.build_messages() — message formatting (no API call)
6. ClaudeClient.estimate_prompt_tokens() — token estimation
7. ClaudeClient.chat() — full flow with mocked SDK

Key principle: ZERO real API calls in this file.
All external dependencies are replaced with unittest.mock.

ISTQB CT-AI relevance:
- 3.1: Unit testing AI components in isolation
- 4.1: Test oracles for AI systems
- 5.2: Mocking non-deterministic components
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pydantic import ValidationError

from src.ai_client.claude_client import (
    ClaudeClient,
    ClaudeConfig,
    ClaudeResponse,
    ChatMessage,
    UsageTracker,
)
from src.ai_client.validators import (
    validate_min_length,
    validate_max_length,
    validate_word_count,
    validate_contains_keywords,
    validate_no_forbidden_phrases,
    validate_not_empty,
    validate_is_valid_json,
    validate_json_has_keys,
    validate_matches_pattern,
    run_validators,
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ClaudeConfig validation
# ═══════════════════════════════════════════════════════════════

class TestClaudeConfig:
    """Tests for configuration validation.
    
    Why test config? Misconfigured AI clients are a common source of
    production bugs. Catching them early is cheap.
    """

    def test_default_config_uses_env_defaults(self):
        """Config should read from environment with sensible defaults."""
        config = ClaudeConfig(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        assert config.max_tokens == 1024
        assert config.timeout == 30.0

    def test_model_must_start_with_claude(self):
        """Non-claude model strings should be rejected immediately."""
        with pytest.raises(ValidationError) as exc_info:
            ClaudeConfig(api_key="sk-test", model="gpt-4o")
        assert "Unknown model" in str(exc_info.value)

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            ClaudeConfig(api_key="sk-test", model="claude-sonnet-4-20250514", max_tokens=0)

    def test_max_tokens_must_not_exceed_limit(self):
        with pytest.raises(ValidationError):
            ClaudeConfig(api_key="sk-test", model="claude-sonnet-4-20250514", max_tokens=99999)

    def test_valid_config_passes(self):
        config = ClaudeConfig(
            api_key="sk-ant-test123",
            model="claude-sonnet-4-20250514",
            max_tokens=512,
        )
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 512


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ChatMessage validation
# ═══════════════════════════════════════════════════════════════

class TestChatMessage:

    def test_valid_user_message(self):
        msg = ChatMessage(role="user", content="Hello Claude")
        assert msg.role == "user"
        assert msg.content == "Hello Claude"

    def test_valid_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hello! How can I help?")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="system", content="You are helpful")
        assert "user" in str(exc_info.value)

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_whitespace_only_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="   \n\t  ")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — ClaudeResponse computed properties
# ═══════════════════════════════════════════════════════════════

class TestClaudeResponse:
    """Test the response model's computed properties.
    
    These are pure computations — no API needed.
    """

    @pytest.fixture
    def sample_response(self) -> ClaudeResponse:
        return ClaudeResponse(
            text="The capital of France is Paris.",
            input_tokens=15,
            output_tokens=8,
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
            latency_ms=342.5,
        )

    def test_total_tokens(self, sample_response):
        assert sample_response.total_tokens == 23

    def test_is_complete_when_end_turn(self, sample_response):
        assert sample_response.is_complete() is True

    def test_is_not_complete_when_max_tokens(self):
        resp = ClaudeResponse(
            text="This response was cut",
            input_tokens=10,
            output_tokens=1024,
            stop_reason="max_tokens",
            model="claude-sonnet-4-20250514",
            latency_ms=5000.0,
        )
        assert resp.is_complete() is False

    def test_estimated_cost_is_non_negative(self, sample_response):
        assert sample_response.estimated_cost_usd >= 0

    def test_cost_scales_with_tokens(self):
        cheap = ClaudeResponse(
            text="hi", input_tokens=10, output_tokens=5,
            stop_reason="end_turn", model="claude-sonnet-4-20250514", latency_ms=100,
        )
        expensive = ClaudeResponse(
            text="hi", input_tokens=100_000, output_tokens=50_000,
            stop_reason="end_turn", model="claude-sonnet-4-20250514", latency_ms=100,
        )
        assert expensive.estimated_cost_usd > cheap.estimated_cost_usd


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — UsageTracker
# ═══════════════════════════════════════════════════════════════

class TestUsageTracker:

    def test_starts_empty(self):
        tracker = UsageTracker()
        assert tracker.call_count == 0
        assert tracker.total_cost_usd == 0.0

    def test_records_responses(self):
        tracker = UsageTracker()
        resp = ClaudeResponse(
            text="test", input_tokens=100, output_tokens=50,
            stop_reason="end_turn", model="claude-sonnet-4-20250514", latency_ms=200,
        )
        tracker.record(resp)
        assert tracker.call_count == 1
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50

    def test_accumulates_across_calls(self):
        tracker = UsageTracker()
        for _ in range(5):
            resp = ClaudeResponse(
                text="x", input_tokens=10, output_tokens=5,
                stop_reason="end_turn", model="claude-sonnet-4-20250514", latency_ms=100,
            )
            tracker.record(resp)
        assert tracker.call_count == 5
        assert tracker.total_input_tokens == 50

    def test_summary_dict_has_expected_keys(self):
        tracker = UsageTracker()
        summary = tracker.summary()
        assert "calls" in summary
        assert "estimated_cost_usd" in summary
        assert "total_tokens" in summary


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — ClaudeClient (no API calls)
# ═══════════════════════════════════════════════════════════════

class TestClaudeClientNoApi:
    """Tests for client logic that doesn't require API calls."""

    @pytest.fixture
    def client(self) -> ClaudeClient:
        """Create client with a fake key — mocked so no real calls happen."""
        config = ClaudeConfig(
            api_key="sk-ant-fake-key-for-unit-tests",
            model="claude-sonnet-4-20250514",
        )
        return ClaudeClient(config=config)

    def test_build_messages_formats_correctly(self, client):
        """build_messages must produce exactly what the SDK expects."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?"),
        ]
        result = client.build_messages(messages)
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

    def test_build_messages_empty_list(self, client):
        assert client.build_messages([]) == []

    def test_estimate_prompt_tokens_returns_positive(self, client):
        estimate = client.estimate_prompt_tokens("Hello world this is a test")
        assert estimate > 0

    def test_estimate_prompt_tokens_scales_with_length(self, client):
        short = client.estimate_prompt_tokens("Hi")
        long = client.estimate_prompt_tokens("Hi " * 100)
        assert long > short

    @patch("src.ai_client.claude_client.anthropic.Anthropic")
    def test_chat_calls_sdk_with_correct_params(self, mock_anthropic_class):
        """Verify the client passes our config values to the SDK correctly."""
        # Arrange: set up mock SDK response
        mock_sdk_instance = MagicMock()
        mock_anthropic_class.return_value = mock_sdk_instance

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Paris")]
        mock_message.usage.input_tokens = 10
        mock_message.usage.output_tokens = 5
        mock_message.stop_reason = "end_turn"
        mock_message.model = "claude-sonnet-4-20250514"
        mock_sdk_instance.messages.create.return_value = mock_message

        config = ClaudeConfig(api_key="sk-ant-fake", model="claude-sonnet-4-20250514")
        client = ClaudeClient(config=config)

        # Act
        response = client.chat("What is the capital of France?")

        # Assert SDK was called with our config values
        call_kwargs = mock_sdk_instance.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["messages"] == [{"role": "user", "content": "What is the capital of France?"}]

    @patch("src.ai_client.claude_client.anthropic.Anthropic")
    def test_chat_with_system_prompt(self, mock_anthropic_class):
        """System prompt should be passed through when provided."""
        mock_sdk = MagicMock()
        mock_anthropic_class.return_value = mock_sdk
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="I am a helpful assistant.")]
        mock_msg.usage.input_tokens = 20
        mock_msg.usage.output_tokens = 8
        mock_msg.stop_reason = "end_turn"
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_sdk.messages.create.return_value = mock_msg

        config = ClaudeConfig(api_key="sk-ant-fake", model="claude-sonnet-4-20250514")
        client = ClaudeClient(config=config)

        client.chat("Who are you?", system="You are a helpful assistant.")

        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a helpful assistant."

    @patch("src.ai_client.claude_client.anthropic.Anthropic")
    def test_chat_returns_typed_response(self, mock_anthropic_class):
        """chat() must return a ClaudeResponse, not a raw dict."""
        mock_sdk = MagicMock()
        mock_anthropic_class.return_value = mock_sdk
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="42")]
        mock_msg.usage.input_tokens = 5
        mock_msg.usage.output_tokens = 2
        mock_msg.stop_reason = "end_turn"
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_sdk.messages.create.return_value = mock_msg

        config = ClaudeConfig(api_key="sk-ant-fake", model="claude-sonnet-4-20250514")
        client = ClaudeClient(config=config)

        response = client.chat("What is 6 x 7?")

        assert isinstance(response, ClaudeResponse)
        assert response.text == "42"
        assert response.input_tokens == 5
        assert response.output_tokens == 2
        assert response.stop_reason == "end_turn"
        assert response.latency_ms >= 0

    @patch("src.ai_client.claude_client.anthropic.Anthropic")
    def test_usage_tracker_updates_after_chat(self, mock_anthropic_class):
        """After a call, usage tracker must reflect the token counts."""
        mock_sdk = MagicMock()
        mock_anthropic_class.return_value = mock_sdk
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Hello!")]
        mock_msg.usage.input_tokens = 25
        mock_msg.usage.output_tokens = 10
        mock_msg.stop_reason = "end_turn"
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_sdk.messages.create.return_value = mock_msg

        config = ClaudeConfig(api_key="sk-ant-fake", model="claude-sonnet-4-20250514")
        client = ClaudeClient(config=config)

        assert client.usage.call_count == 0
        client.chat("Hi")
        assert client.usage.call_count == 1
        assert client.usage.total_input_tokens == 25


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — Validators
# ═══════════════════════════════════════════════════════════════

class TestValidators:
    """Tests for every validator function.
    
    Validators are pure functions — trivial to test exhaustively.
    These tests also serve as living documentation of validator behaviour.
    """

    # ── Length validators ───────────────────────────

    def test_min_length_passes(self):
        result = validate_min_length("Hello world!", min_chars=5)
        assert result.passed is True

    def test_min_length_fails(self):
        result = validate_min_length("Hi", min_chars=10)
        assert result.passed is False
        assert "10" in result.message

    def test_max_length_passes(self):
        result = validate_max_length("Short text", max_chars=100)
        assert result.passed is True

    def test_max_length_fails(self):
        result = validate_max_length("A" * 200, max_chars=100)
        assert result.passed is False

    def test_word_count_within_range(self):
        result = validate_word_count("one two three four five", min_words=3, max_words=10)
        assert result.passed is True

    def test_word_count_too_few(self):
        result = validate_word_count("hi", min_words=5, max_words=10)
        assert result.passed is False

    def test_word_count_too_many(self):
        result = validate_word_count("one " * 20, min_words=1, max_words=10)
        assert result.passed is False

    def test_word_count_exactly_five_words(self):
        # "one two three four five" = exactly 5 words
        # Range 3-7 includes 5 → passes
        result = validate_word_count("one two three four five", min_words=3, max_words=7)
        assert result.passed is True

        # Range 6-10 excludes 5 → fails (5 < 6)
        result = validate_word_count("one two three four five", min_words=6, max_words=10)
        assert result.passed is False                    


    # ── Content validators ──────────────────────────

    def test_contains_keywords_all_present(self):
        result = validate_contains_keywords(
            "Python is great for testing AI systems",
            keywords=["python", "testing", "ai"],
        )
        assert result.passed is True

    def test_contains_keywords_one_missing(self):
        result = validate_contains_keywords(
            "Python is great for testing",
            keywords=["python", "testing", "java"],
        )
        assert result.passed is False
        assert "java" in str(result.details["missing"])

    def test_contains_keywords_case_insensitive_by_default(self):
        result = validate_contains_keywords("PYTHON is great", keywords=["python"])
        assert result.passed is True

    def test_contains_keywords_case_sensitive(self):
        result = validate_contains_keywords(
            "PYTHON is great", keywords=["python"], case_sensitive=True
        )
        assert result.passed is False

    def test_no_forbidden_phrases_clean_text(self):
        result = validate_no_forbidden_phrases(
            "This is a helpful response.",
            forbidden=["harmful", "dangerous", "illegal"],
        )
        assert result.passed is True

    def test_no_forbidden_phrases_detects_violation(self):
        result = validate_no_forbidden_phrases(
            "This contains harmful content",
            forbidden=["harmful", "dangerous"],
        )
        assert result.passed is False
        assert "harmful" in result.details["found"]

    def test_not_empty_passes(self):
        assert validate_not_empty("Hello world").passed is True

    def test_not_empty_fails_on_whitespace(self):
        assert validate_not_empty("   \n  ").passed is False

    def test_not_empty_fails_on_empty_string(self):
        assert validate_not_empty("").passed is False

    # ── JSON validators ─────────────────────────────

    def test_valid_json_object(self):
        result = validate_is_valid_json('{"name": "Alice", "age": 30}')
        assert result.passed is True

    def test_valid_json_array(self):
        result = validate_is_valid_json('[1, 2, 3]')
        assert result.passed is True

    def test_invalid_json_detected(self):
        result = validate_is_valid_json("This is not JSON at all")
        assert result.passed is False
        assert "Invalid JSON" in result.message

    def test_json_has_required_keys_all_present(self):
        json_str = '{"status": "ok", "data": [], "count": 0}'
        result = validate_json_has_keys(json_str, required_keys=["status", "data", "count"])
        assert result.passed is True

    def test_json_has_required_keys_missing_key(self):
        json_str = '{"status": "ok"}'
        result = validate_json_has_keys(json_str, required_keys=["status", "data"])
        assert result.passed is False
        assert "data" in result.details["missing"]

    def test_json_has_keys_fails_gracefully_on_invalid_json(self):
        result = validate_json_has_keys("not json", required_keys=["key"])
        assert result.passed is False

    # ── Pattern validators ──────────────────────────

    def test_matches_pattern_found(self):
        result = validate_matches_pattern("Order #12345 confirmed", r"Order #\d+")
        assert result.passed is True

    def test_matches_pattern_not_found(self):
        result = validate_matches_pattern("No order here", r"Order #\d+")
        assert result.passed is False


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — Composite ValidationReport
# ═══════════════════════════════════════════════════════════════

class TestValidationReport:
    """Tests for the composite validation runner."""

    def test_all_pass(self):
        report = run_validators(
            validate_not_empty("Hello"),
            validate_min_length("Hello world", 5),
        )
        assert report.passed is True
        assert len(report.failures) == 0

    def test_one_failure_fails_report(self):
        report = run_validators(
            validate_not_empty("Hello"),
            validate_min_length("Hi", 100),  # will fail
        )
        assert report.passed is False
        assert len(report.failures) == 1

    def test_summary_lists_failures(self):
        report = run_validators(
            validate_min_length("Hi", 100),
            validate_max_length("A" * 999, 10),
        )
        summary = report.summary()
        assert "2/2 checks failed" in summary
        assert "min_length" in summary
        assert "max_length" in summary

    def test_summary_success_message(self):
        report = run_validators(validate_not_empty("Hello"))
        assert "passed" in report.summary()