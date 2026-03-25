"""
validators.py — Reusable response validators for AI output quality testing.

These are the building blocks of your AI test assertions.
Each validator is a pure function: input → bool/ValidationResult.
Pure functions are trivial to unit-test.

ISTQB CT-AI relevance:
- Section 3.3: Testing AI output quality
- Section 4.2: Defining oracles for non-deterministic systems
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    check_name: str
    message: str
    details: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.passed


# ─────────────────────────────────────────────
# Length validators
# ─────────────────────────────────────────────

def validate_min_length(text: str, min_chars: int) -> ValidationResult:
    """Response must have at least min_chars characters."""
    actual = len(text.strip())
    passed = actual >= min_chars
    return ValidationResult(
        passed=passed,
        check_name="min_length",
        message=f"Expected >={min_chars} chars, got {actual}" if not passed else "OK",
        details={"actual": actual, "minimum": min_chars},
    )


def validate_max_length(text: str, max_chars: int) -> ValidationResult:
    """Response must not exceed max_chars characters."""
    actual = len(text.strip())
    passed = actual <= max_chars
    return ValidationResult(
        passed=passed,
        check_name="max_length",
        message=f"Expected <={max_chars} chars, got {actual}" if not passed else "OK",
        details={"actual": actual, "maximum": max_chars},
    )


def validate_word_count(text: str, min_words: int, max_words: int) -> ValidationResult:
    """Response word count must be within [min_words, max_words]."""
    words = len(text.split())
    passed = min_words <= words <= max_words
    return ValidationResult(
        passed=passed,
        check_name="word_count",
        message=f"Expected {min_words}–{max_words} words, got {words}" if not passed else "OK",
        details={"actual_words": words, "min": min_words, "max": max_words},
    )


# ─────────────────────────────────────────────
# Content validators
# ─────────────────────────────────────────────

def validate_contains_keywords(
    text: str,
    keywords: list[str],
    require_all: bool = True,
    case_sensitive: bool = False,
) -> ValidationResult:
    """Response must contain the specified keywords."""
    check_text = text if case_sensitive else text.lower()
    check_words = keywords if case_sensitive else [k.lower() for k in keywords]

    found = [kw for kw in check_words if kw in check_text]
    missing = [kw for kw in check_words if kw not in check_text]

    if require_all:
        passed = len(missing) == 0
        message = f"Missing keywords: {missing}" if not passed else "OK"
    else:
        passed = len(found) > 0
        message = "No keywords found" if not passed else f"Found: {found}"

    return ValidationResult(
        passed=passed,
        check_name="contains_keywords",
        message=message,
        details={"found": found, "missing": missing},
    )


def validate_no_forbidden_phrases(
    text: str,
    forbidden: list[str],
    case_sensitive: bool = False,
) -> ValidationResult:
    """Response must NOT contain any of the forbidden phrases.
    
    Use case: Safety testing — ensure AI doesn't generate harmful content.
    """
    check_text = text if case_sensitive else text.lower()
    found_forbidden = [
        phrase for phrase in forbidden
        if (phrase if case_sensitive else phrase.lower()) in check_text
    ]
    passed = len(found_forbidden) == 0
    return ValidationResult(
        passed=passed,
        check_name="no_forbidden_phrases",
        message=f"Found forbidden phrases: {found_forbidden}" if not passed else "OK",
        details={"found": found_forbidden},
    )


def validate_not_empty(text: str) -> ValidationResult:
    """Response must not be empty or whitespace-only."""
    passed = bool(text.strip())
    return ValidationResult(
        passed=passed,
        check_name="not_empty",
        message="Response is empty" if not passed else "OK",
    )


# ─────────────────────────────────────────────
# Structure validators
# ─────────────────────────────────────────────

def validate_is_valid_json(text: str) -> ValidationResult:
    """Response must be valid JSON (for structured output prompts)."""
    try:
        parsed = json.loads(text.strip())
        return ValidationResult(
            passed=True,
            check_name="is_valid_json",
            message="OK",
            details={"type": type(parsed).__name__},
        )
    except json.JSONDecodeError as exc:
        return ValidationResult(
            passed=False,
            check_name="is_valid_json",
            message=f"Invalid JSON: {exc.msg} at position {exc.pos}",
        )


def validate_json_has_keys(text: str, required_keys: list[str]) -> ValidationResult:
    """JSON response must contain all required top-level keys."""
    json_check = validate_is_valid_json(text)
    if not json_check:
        return ValidationResult(
            passed=False,
            check_name="json_has_keys",
            message=f"Cannot check keys — not valid JSON: {json_check.message}",
        )
    data = json.loads(text.strip())
    if not isinstance(data, dict):
        return ValidationResult(
            passed=False,
            check_name="json_has_keys",
            message=f"Expected JSON object, got {type(data).__name__}",
        )
    missing = [k for k in required_keys if k not in data]
    passed = len(missing) == 0
    return ValidationResult(
        passed=passed,
        check_name="json_has_keys",
        message=f"Missing keys: {missing}" if not passed else "OK",
        details={"present": list(data.keys()), "missing": missing},
    )


def validate_matches_pattern(text: str, pattern: str) -> ValidationResult:
    """Response must match a regex pattern."""
    match = re.search(pattern, text, re.DOTALL)
    passed = match is not None
    return ValidationResult(
        passed=passed,
        check_name="matches_pattern",
        message=f"Pattern not found: {pattern!r}" if not passed else "OK",
        details={"pattern": pattern},
    )


# ─────────────────────────────────────────────
# Composite validator
# ─────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Aggregated results from running multiple validators."""
    results: list[ValidationResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        failed = len(self.failures)
        if failed == 0:
            return f"All {total} checks passed"
        lines = [f"{failed}/{total} checks failed:"]
        for f in self.failures:
            lines.append(f"  ✗ {f.check_name}: {f.message}")
        return "\n".join(lines)


def run_validators(*validators: ValidationResult) -> ValidationReport:
    """Collect multiple validation results into a report."""
    return ValidationReport(results=list(validators))