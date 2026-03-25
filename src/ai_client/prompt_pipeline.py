"""
prompt_pipeline.py — A reusable prompt pipeline component.

This is what Phase 2 tests as a component — not individual functions,
but the PIPELINE as a whole unit of behavior.

A pipeline encapsulates:
- System prompt management
- Input formatting
- Retry logic
- Output parsing
- Quality checks

Think of it as a "prompt function" — a reusable, testable unit
that wraps a specific AI capability.

ISTQB CT-AI relevance:
- 3.3: Component testing of AI pipelines
- 4.3: Testing prompt sensitivity and robustness
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.ai_client.claude_client import ClaudeClient, ClaudeResponse
from src.ai_client.validators import (
    ValidationReport,
    validate_is_valid_json,
    validate_json_has_keys,
    validate_not_empty,
    validate_no_forbidden_phrases,
    run_validators,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Pipeline result
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Result from running a prompt pipeline."""
    success: bool
    response: ClaudeResponse | None
    parsed_data: Any | None          # Parsed JSON if applicable
    validation: ValidationReport | None
    attempts: int
    error: str | None = None

    @property
    def text(self) -> str:
        """Convenience accessor for the response text."""
        return self.response.text if self.response else ""


# ─────────────────────────────────────────────
# Base pipeline
# ─────────────────────────────────────────────

class PromptPipeline:
    """
    Base class for all prompt pipelines.

    Subclass this to create specific AI capabilities:
        class SummaryPipeline(PromptPipeline): ...
        class ClassifierPipeline(PromptPipeline): ...
        class ExtractorPipeline(PromptPipeline): ...
    """

    DEFAULT_SYSTEM = "You are a helpful, accurate, and concise assistant."
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(self, client: ClaudeClient, system_prompt: str | None = None):
        self.client = client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

    def run(self, user_input: str) -> PipelineResult:
        """Run the pipeline with retry logic."""
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.debug("Pipeline attempt %d/%d", attempt, self.MAX_RETRIES)
                response = self.client.chat(
                    self._format_prompt(user_input),
                    system=self.system_prompt,
                )
                parsed = self._parse_response(response)
                validation = self._validate(response, parsed)

                return PipelineResult(
                    success=validation.passed,
                    response=response,
                    parsed_data=parsed,
                    validation=validation,
                    attempts=attempt,
                )

            except Exception as exc:
                last_error = str(exc)
                logger.warning("Attempt %d failed: %s", attempt, exc)
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)

        return PipelineResult(
            success=False,
            response=None,
            parsed_data=None,
            validation=None,
            attempts=self.MAX_RETRIES,
            error=last_error,
        )

    def _format_prompt(self, user_input: str) -> str:
        """Override to transform user input before sending."""
        return user_input

    def _parse_response(self, response: ClaudeResponse) -> Any:
        """Override to parse the response text into structured data."""
        return None

    def _validate(self, response: ClaudeResponse, parsed: Any) -> ValidationReport:
        """Override to add pipeline-specific validation."""
        return run_validators(validate_not_empty(response.text))


# ─────────────────────────────────────────────
# Concrete pipelines
# ─────────────────────────────────────────────

class SummaryPipeline(PromptPipeline):
    """Summarizes text to a target word count."""

    DEFAULT_SYSTEM = (
        "You are a precise summarizer. "
        "Produce concise summaries that preserve the key information. "
        "Never add information not present in the original text."
    )

    def __init__(self, client: ClaudeClient, target_words: int = 50):
        super().__init__(client)
        self.target_words = target_words

    def _format_prompt(self, user_input: str) -> str:
        return (
            f"Summarize the following text in approximately {self.target_words} words.\n\n"
            f"Text to summarize:\n{user_input}"
        )

    def _validate(self, response: ClaudeResponse, parsed: Any) -> ValidationReport:
        min_words = max(1, self.target_words // 2)
        max_words = self.target_words * 2
        words = len(response.text.split())
        from src.ai_client.validators import validate_word_count
        return run_validators(
            validate_not_empty(response.text),
            validate_word_count(response.text, min_words, max_words),
        )


class JsonExtractorPipeline(PromptPipeline):
    """Extracts structured data as JSON with required keys."""

    DEFAULT_SYSTEM = (
        "You are a precise data extractor. "
        "Always respond with ONLY valid JSON — no prose, no markdown, no explanation. "
        "Your entire response must be parseable by json.loads()."
    )

    def __init__(self, client: ClaudeClient, required_keys: list[str]):
        super().__init__(client)
        self.required_keys = required_keys

    def _format_prompt(self, user_input: str) -> str:
        keys_str = ", ".join(f'"{k}"' for k in self.required_keys)
        return (
            f"Extract information and return ONLY a JSON object with these keys: {keys_str}\n\n"
            f"Input:\n{user_input}"
        )

    def _parse_response(self, response: ClaudeResponse) -> dict | None:
        """Attempt to parse response as JSON."""
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _validate(self, response: ClaudeResponse, parsed: Any) -> ValidationReport:
        return run_validators(
            validate_not_empty(response.text),
            validate_is_valid_json(response.text.strip()),
            validate_json_has_keys(response.text.strip(), self.required_keys),
        )


class SafetyCheckerPipeline(PromptPipeline):
    """Pipeline that validates AI responses don't contain harmful content."""

    FORBIDDEN = [
        "step-by-step instructions",
        "here's how to",
        "detailed guide",
    ]

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. "
        "Never provide instructions for harmful, illegal, or dangerous activities. "
        "If asked about such topics, explain that you cannot help."
    )

    def _validate(self, response: ClaudeResponse, parsed: Any) -> ValidationReport:
        return run_validators(
            validate_not_empty(response.text),
            validate_no_forbidden_phrases(response.text, self.FORBIDDEN),
        )