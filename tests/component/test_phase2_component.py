"""
tests/component/test_phase2_component.py

Phase 2: Component Testing

What changes from Phase 1:
- We test COMPONENTS (pipeline = multiple classes working together)
- We use shared fixtures from conftest.py instead of local mocks
- We test BEHAVIOR ("does the pipeline retry on failure?")
  not just implementation ("did this function get called?")
- We test prompt sensitivity — small input changes, stable outputs

Key insight: Component tests find bugs that unit tests miss.
A unit test can verify ClaudeClient.chat() works AND
SummaryPipeline._format_prompt() works, but only a component
test verifies they work TOGETHER correctly.

ISTQB CT-AI relevance:
- 3.3: Component testing of AI pipelines
- 4.3: Testing non-deterministic components
- 5.3: Metamorphic testing for AI systems
"""

import pytest
from unittest.mock import MagicMock, patch, call
import json

from src.ai_client.claude_client import ClaudeClient, ClaudeConfig, ClaudeResponse
from src.ai_client.prompt_pipeline import (
    PromptPipeline,
    SummaryPipeline,
    JsonExtractorPipeline,
    SafetyCheckerPipeline,
    PipelineResult,
)

# Import the helper from conftest (pytest makes it available automatically)
# but we import explicitly here for clarity in tests
from conftest import make_mock_sdk_message


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — Fixtures (local to this file)
# These complement the shared fixtures in conftest.py
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def summary_pipeline(mock_client) -> SummaryPipeline:
    """A SummaryPipeline wired to the mocked client from conftest."""
    return SummaryPipeline(client=mock_client, target_words=30)


@pytest.fixture
def json_pipeline(mock_client) -> JsonExtractorPipeline:
    """A JsonExtractorPipeline with required keys."""
    return JsonExtractorPipeline(
        client=mock_client,
        required_keys=["name", "age", "city"],
    )


@pytest.fixture
def safety_pipeline(mock_client) -> SafetyCheckerPipeline:
    """A SafetyCheckerPipeline wired to the mocked client."""
    return SafetyCheckerPipeline(client=mock_client)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — Base Pipeline behavior
# ═══════════════════════════════════════════════════════════════

class TestBasePipeline:
    """Tests for shared pipeline behavior — retry, error handling, result shape."""

    def test_successful_run_returns_pipeline_result(self, mock_client):
        """A successful run must return a PipelineResult, not a raw response."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Hello world this is a response"
        )
        pipeline = PromptPipeline(client=mock_client)
        result = pipeline.run("Say hello")

        assert isinstance(result, PipelineResult)
        assert result.response is not None
        assert result.attempts == 1

    def test_successful_run_records_attempt_count(self, mock_client):
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Success on first try"
        )
        pipeline = PromptPipeline(client=mock_client)
        result = pipeline.run("Test")
        assert result.attempts == 1

    def test_pipeline_retries_on_exception(self, mock_client):
        """Pipeline must retry up to MAX_RETRIES times on API failure.
        
        This tests resilience — critical for production AI systems
        where transient failures are common.
        """
        # First two calls raise, third succeeds
        mock_client._mock_sdk.messages.create.side_effect = [
            Exception("Network timeout"),
            Exception("Rate limit"),
            make_mock_sdk_message(text="Finally succeeded"),
        ]

        pipeline = PromptPipeline(client=mock_client)
        # Disable sleep delay for tests
        pipeline.RETRY_DELAY = 0
        result = pipeline.run("Test prompt")

        assert result.attempts == 3
        assert result.error is None

    def test_pipeline_returns_failure_after_max_retries(self, mock_client):
        """After MAX_RETRIES failures, pipeline returns a failure result — not an exception."""
        mock_client._mock_sdk.messages.create.side_effect = Exception("Persistent failure")

        pipeline = PromptPipeline(client=mock_client)
        pipeline.RETRY_DELAY = 0
        result = pipeline.run("Test")

        assert result.success is False
        assert result.error is not None
        assert result.response is None
        assert result.attempts == pipeline.MAX_RETRIES

    def test_failed_result_has_no_parsed_data(self, mock_client):
        mock_client._mock_sdk.messages.create.side_effect = Exception("Error")
        pipeline = PromptPipeline(client=mock_client)
        pipeline.RETRY_DELAY = 0
        result = pipeline.run("Test")
        assert result.parsed_data is None

    def test_result_text_property_on_success(self, mock_client):
        """PipelineResult.text should proxy to response.text."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="The answer is 42"
        )
        pipeline = PromptPipeline(client=mock_client)
        result = pipeline.run("What is the answer?")
        assert result.text == "The answer is 42"

    def test_result_text_property_on_failure(self, mock_client):
        """PipelineResult.text should return empty string when no response."""
        mock_client._mock_sdk.messages.create.side_effect = Exception("Error")
        pipeline = PromptPipeline(client=mock_client)
        pipeline.RETRY_DELAY = 0
        result = pipeline.run("Test")
        assert result.text == ""


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — SummaryPipeline component tests
# ═══════════════════════════════════════════════════════════════

class TestSummaryPipeline:
    """Tests for the SummaryPipeline component."""

    def test_prompt_includes_target_word_count(self, mock_client):
        """The formatted prompt must tell the model exactly how many words to use."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="This is a concise summary of the text provided above."
        )
        pipeline = SummaryPipeline(client=mock_client, target_words=30)
        pipeline.run("Some long text to summarize...")

        actual_prompt = mock_client._mock_sdk.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "30" in actual_prompt
        assert "summarize" in actual_prompt.lower()

    def test_prompt_includes_user_input(self, mock_client):
        """The user's text must appear in the formatted prompt."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="A brief summary."
        )
        pipeline = SummaryPipeline(client=mock_client, target_words=20)
        user_text = "The quick brown fox jumps over the lazy dog."
        pipeline.run(user_text)

        actual_prompt = mock_client._mock_sdk.messages.create.call_args.kwargs["messages"][0]["content"]
        assert user_text in actual_prompt

    def test_uses_summary_system_prompt(self, mock_client):
        """SummaryPipeline must use its own system prompt, not the base default."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Summary here."
        )
        pipeline = SummaryPipeline(client=mock_client, target_words=20)
        pipeline.run("Text to summarize")

        system = mock_client._mock_sdk.messages.create.call_args.kwargs.get("system", "")
        assert "summarizer" in system.lower()

    def test_validation_passes_for_appropriate_length(self, mock_client):
        """A response within word count range should pass validation."""
        # target=30, so valid range is 15-60 words
        summary_text = " ".join(["word"] * 30)  # exactly 30 words
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text=summary_text
        )
        pipeline = SummaryPipeline(client=mock_client, target_words=30)
        result = pipeline.run("Summarize this.")
        assert result.success is True

    def test_validation_fails_for_single_word_response(self, mock_client):
        """A one-word summary should fail validation (too short)."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Yes."
        )
        pipeline = SummaryPipeline(client=mock_client, target_words=30)
        result = pipeline.run("Summarize this long document...")
        assert result.success is False

    def test_different_target_words_changes_prompt(self, mock_client):
        """Changing target_words must produce a different prompt — prompt sensitivity test."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text=" ".join(["word"] * 50)
        )

        pipeline_50 = SummaryPipeline(client=mock_client, target_words=50)
        pipeline_50.run("Some text")
        prompt_50 = mock_client._mock_sdk.messages.create.call_args.kwargs["messages"][0]["content"]

        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text=" ".join(["word"] * 100)
        )
        pipeline_100 = SummaryPipeline(client=mock_client, target_words=100)
        pipeline_100.run("Some text")
        prompt_100 = mock_client._mock_sdk.messages.create.call_args.kwargs["messages"][0]["content"]

        assert prompt_50 != prompt_100
        assert "50" in prompt_50
        assert "100" in prompt_100


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — JsonExtractorPipeline component tests
# ═══════════════════════════════════════════════════════════════

class TestJsonExtractorPipeline:
    """Tests for the JSON extraction pipeline."""

    def test_valid_json_response_is_parsed(self, mock_client):
        """A valid JSON response must be parsed into a dict."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text='{"name": "Alice", "age": 30, "city": "Paris"}'
        )
        pipeline = JsonExtractorPipeline(
            client=mock_client,
            required_keys=["name", "age", "city"],
        )
        result = pipeline.run("Extract: Alice is 30 years old and lives in Paris.")

        assert result.success is True
        assert result.parsed_data is not None
        assert result.parsed_data["name"] == "Alice"
        assert result.parsed_data["city"] == "Paris"

    def test_invalid_json_fails_validation(self, mock_client):
        """A non-JSON response must fail validation."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Alice is 30 years old and lives in Paris."
        )
        pipeline = JsonExtractorPipeline(
            client=mock_client,
            required_keys=["name", "age", "city"],
        )
        result = pipeline.run("Extract info from: Alice, 30, Paris")
        assert result.success is False

    def test_json_missing_required_key_fails(self, mock_client):
        """JSON missing a required key must fail validation."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text='{"name": "Alice", "city": "Paris"}'  # missing "age"
        )
        pipeline = JsonExtractorPipeline(
            client=mock_client,
            required_keys=["name", "age", "city"],
        )
        result = pipeline.run("Extract: Alice lives in Paris.")
        assert result.success is False
        assert "age" in result.validation.summary()

    def test_prompt_includes_required_keys(self, mock_client):
        """The formatted prompt must list the required JSON keys."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text='{"name": "Bob", "age": 25, "city": "London"}'
        )
        pipeline = JsonExtractorPipeline(
            client=mock_client,
            required_keys=["name", "age", "city"],
        )
        pipeline.run("Bob is 25 and lives in London.")

        prompt = mock_client._mock_sdk.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "name" in prompt
        assert "age" in prompt
        assert "city" in prompt

    def test_strips_markdown_code_fences(self, mock_client):
        """Pipeline must handle responses wrapped in markdown code fences.
        
        Claude sometimes wraps JSON in ```json ... ``` even when told not to.
        Robust pipelines handle this gracefully.
        """
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text='```json\n{"name": "Alice", "age": 30, "city": "Paris"}\n```'
        )
        pipeline = JsonExtractorPipeline(
            client=mock_client,
            required_keys=["name", "age", "city"],
        )
        result = pipeline.run("Extract: Alice, 30, Paris")
        assert result.parsed_data is not None
        assert result.parsed_data["name"] == "Alice"

    def test_different_required_keys_changes_prompt(self, mock_client):
        """Metamorphic test: changing required keys must change the prompt."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text='{"title": "Engineer", "company": "Acme"}'
        )
        pipeline_job = JsonExtractorPipeline(
            client=mock_client,
            required_keys=["title", "company"],
        )
        pipeline_job.run("She is an Engineer at Acme")
        prompt_job = mock_client._mock_sdk.messages.create.call_args.kwargs["messages"][0]["content"]

        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text='{"name": "Alice", "age": 30, "city": "Paris"}'
        )
        pipeline_person = JsonExtractorPipeline(
            client=mock_client,
            required_keys=["name", "age", "city"],
        )
        pipeline_person.run("Alice is 30, lives in Paris")
        prompt_person = mock_client._mock_sdk.messages.create.call_args.kwargs["messages"][0]["content"]

        assert prompt_job != prompt_person


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — SafetyCheckerPipeline component tests
# ═══════════════════════════════════════════════════════════════

class TestSafetyCheckerPipeline:
    """Tests for safety validation in the pipeline.
    
    Safety testing is a core requirement in ISTQB CT-AI.
    These tests verify the AI system doesn't produce harmful outputs.
    """

    def test_safe_response_passes(self, mock_client):
        """A normal, safe response must pass safety validation."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="I'm sorry, I can't help with that request."
        )
        pipeline = SafetyCheckerPipeline(client=mock_client)
        result = pipeline.run("How do I make a bomb?")
        assert result.success is True

    def test_response_with_harmful_instructions_fails(self, mock_client):
        """A response containing step-by-step harmful instructions must fail."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Here's how to do it: step-by-step instructions follow..."
        )
        pipeline = SafetyCheckerPipeline(client=mock_client)
        result = pipeline.run("Dangerous request")
        assert result.success is False

    def test_safety_system_prompt_is_set(self, mock_client):
        """Safety pipeline must use its specific system prompt."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="I cannot help with that."
        )
        pipeline = SafetyCheckerPipeline(client=mock_client)
        pipeline.run("Some request")

        system = mock_client._mock_sdk.messages.create.call_args.kwargs.get("system", "")
        assert "harmful" in system.lower() or "dangerous" in system.lower()


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — Cross-pipeline / integration of component pieces
# ═══════════════════════════════════════════════════════════════

class TestPipelineWithConftest:
    """Tests that use shared fixtures from conftest.py.
    
    These demonstrate how conftest fixtures reduce boilerplate
    and keep tests DRY across the test suite.
    """

    def test_pipeline_uses_sample_prompts(self, mock_client, sample_prompts):
        """Use the shared sample_prompts fixture from conftest."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Paris is the capital of France."
        )
        pipeline = PromptPipeline(client=mock_client)
        result = pipeline.run(sample_prompts["factual"])
        assert result.response is not None

    def test_pipeline_result_text_matches_mock(self, mock_client, sample_prompts):
        expected_text = "The capital of France is Paris."
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text=expected_text
        )
        pipeline = PromptPipeline(client=mock_client)
        result = pipeline.run(sample_prompts["factual"])
        assert result.text == expected_text

    def test_usage_tracker_increments_across_pipeline_calls(self, mock_client):
        """Usage tracker must count all calls made through a pipeline."""
        mock_client._mock_sdk.messages.create.return_value = make_mock_sdk_message(
            text="Response"
        )
        pipeline = PromptPipeline(client=mock_client)

        for _ in range(3):
            pipeline.run("Test prompt")

        assert mock_client.usage.call_count == 3