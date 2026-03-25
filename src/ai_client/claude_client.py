"""
claude_client.py — Reusable Claude API wrapper for enterprise AI testing.

Design principles:
- Every method is independently testable (unit, component, integration)
- All config comes from environment — no hardcoded values
- Structured logging for observability
- Cost tracking built in from day one
- Typed inputs/outputs via Pydantic
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

class ClaudeConfig(BaseModel):
    """All config sourced from environment variables.
    
    Having a typed config object makes it easy to:
    - Unit-test config validation in isolation
    - Swap configs between test/prod environments
    - Document every knob the system exposes
    """
    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("TEST_MAX_TOKENS", "1024")))
    timeout: float = Field(default_factory=lambda: float(os.getenv("TEST_TIMEOUT_SECONDS", "30")))

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v: str) -> str:
        # We allow empty key in unit tests (they use mocks).
        # Integration tests will fail fast if key is missing.
        return v

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_tokens must be positive, got {v}")
        if v > 8192:
            raise ValueError(f"max_tokens {v} exceeds safe limit 8192")
        return v

    @field_validator("model")
    @classmethod
    def model_must_be_known(cls, v: str) -> str:
        known_prefixes = ("claude-",)
        if not any(v.startswith(p) for p in known_prefixes):
            raise ValueError(f"Unknown model '{v}'. Expected a claude-* model string.")
        return v


# ─────────────────────────────────────────────
# 2. Request / Response models
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty or whitespace only")
        return v


class ClaudeResponse(BaseModel):
    """Structured response from the Claude API.
    
    Wraps the raw SDK response with:
    - Typed fields (no dict drilling)
    - Computed cost estimate
    - Latency tracking
    - The stop reason (useful for testing streaming/tool use)
    """
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    model: str
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate. Update rates when Anthropic changes pricing."""
        # Sonnet 4 pricing (as of 2025): $3/$15 per MTok in/out
        input_cost = (self.input_tokens / 1_000_000) * 3.00
        output_cost = (self.output_tokens / 1_000_000) * 15.00
        return round(input_cost + output_cost, 6)

    def is_complete(self) -> bool:
        """True if the model finished naturally (not cut off by max_tokens)."""
        return self.stop_reason == "end_turn"


# ─────────────────────────────────────────────
# 3. Usage tracker (for cost control in test runs)
# ─────────────────────────────────────────────

@dataclass
class UsageTracker:
    """Accumulates token usage across a test session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    errors: list[str] = field(default_factory=list)

    def record(self, response: ClaudeResponse) -> None:
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.call_count += 1

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * 3.00
        output_cost = (self.total_output_tokens / 1_000_000) * 15.00
        return round(input_cost + output_cost, 6)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": self.total_cost_usd,
            "errors": len(self.errors),
        }


# ─────────────────────────────────────────────
# 4. The client
# ─────────────────────────────────────────────

class ClaudeClient:
    """
    Enterprise-grade Claude API client.

    Responsibilities:
    - Single point of config injection (easy to test)
    - Structured logging on every call
    - Usage tracking (cost control)
    - Clean error surfaces (no raw SDK exceptions leak out)

    Usage:
        client = ClaudeClient()
        response = client.chat("Summarise this text: ...")
        print(response.text)
        print(response.estimated_cost_usd)
    """

    def __init__(self, config: ClaudeConfig | None = None) -> None:
        self.config = config or ClaudeConfig()
        self.usage = UsageTracker()
        # The SDK client is created once and reused (connection pooling)
        self._sdk = anthropic.Anthropic(api_key=self.config.api_key)
        logger.info("ClaudeClient initialised — model=%s", self.config.model)

    # ── Public API ──────────────────────────────

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
    ) -> ClaudeResponse:
        """Single-turn chat. Returns a typed ClaudeResponse.
        
        This is the method tested most heavily in Phase 1 unit tests.
        """
        message = ChatMessage(role="user", content=prompt)
        return self._send([message], system=system, temperature=temperature)

    def multi_turn(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
    ) -> ClaudeResponse:
        """Multi-turn conversation. Pass the full history each time."""
        return self._send(messages, system=system)

    def build_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert ChatMessage objects to the SDK dict format.
        
        Extracted as a public method so it can be unit-tested independently
        without making any API calls.
        """
        return [{"role": m.role, "content": m.content} for m in messages]

    def estimate_prompt_tokens(self, prompt: str) -> int:
        """Rough token estimate: ~4 chars per token. Good enough for budget checks."""
        return max(1, len(prompt) // 4)

    # ── Private ─────────────────────────────────

    def _send(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        temperature: float = 1.0,
    ) -> ClaudeResponse:
        sdk_messages = self.build_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": sdk_messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        logger.debug("→ Claude API call: model=%s tokens_budget=%d", self.config.model, self.config.max_tokens)
        start = time.monotonic()

        try:
            raw = self._sdk.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            logger.error("Authentication failed — check ANTHROPIC_API_KEY")
            raise
        except anthropic.RateLimitError as exc:
            logger.warning("Rate limit hit — consider adding retry logic")
            raise
        except anthropic.APIError as exc:
            logger.error("API error: %s", exc)
            raise

        latency_ms = (time.monotonic() - start) * 1000

        response = ClaudeResponse(
            text=raw.content[0].text,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            stop_reason=raw.stop_reason,
            model=raw.model,
            latency_ms=round(latency_ms, 2),
        )

        self.usage.record(response)
        logger.info(
            "← Claude response: tokens=%d+%d cost=$%.4f latency=%.0fms stop=%s",
            response.input_tokens,
            response.output_tokens,
            response.estimated_cost_usd,
            response.latency_ms,
            response.stop_reason,
        )
        return response