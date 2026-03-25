"""
src/ai_client/eval_harness.py

LLM Evaluation Harness — measures AI output quality at system level.

This is the core of Phase 4 system testing. It answers:
"How GOOD is our AI system, not just does it work?"

Key concepts:
- EvalCase:    one test case (input + expected behavior)
- EvalResult:  one scored result (0.0 to 1.0)
- EvalSuite:   collection of cases run together
- Scorer:      function that scores a response

ISTQB CT-AI relevance:
- 3.5: System testing of AI systems
- 4.5: AI-specific quality metrics
- 5.5: Evaluation-based testing
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum

from src.ai_client.claude_client import ClaudeClient, ClaudeResponse


# ─────────────────────────────────────────────
# 1. Quality dimensions
# ─────────────────────────────────────────────

class QualityDimension(Enum):
    """
    The dimensions we measure AI quality on.
    Based on ISTQB CT-AI quality characteristics.
    """
    CORRECTNESS   = "correctness"    # Is the answer factually right?
    COMPLETENESS  = "completeness"   # Does it cover everything asked?
    CONCISENESS   = "conciseness"    # Is it appropriately brief?
    SAFETY        = "safety"         # Does it avoid harmful content?
    FORMAT        = "format"         # Does it match the required format?
    CONSISTENCY   = "consistency"    # Same quality across multiple runs?


# ─────────────────────────────────────────────
# 2. Core data structures
# ─────────────────────────────────────────────

@dataclass
class EvalCase:
    """
    One evaluation test case.

    Think of this as a row in a test dataset:
    - What do we send to the AI?
    - What do we expect back?
    - How do we score the response?
    """
    id: str                              # Unique identifier
    prompt: str                          # Input to the AI
    system_prompt: str | None = None     # Optional system context
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    expected_format: str | None = None   # "json", "list", "paragraph"
    dimension: QualityDimension = QualityDimension.CORRECTNESS
    tags: list[str] = field(default_factory=list)  # For filtering


@dataclass
class EvalResult:
    """
    Result of running one EvalCase.
    Score is 0.0 (complete failure) to 1.0 (perfect).
    """
    case_id: str
    score: float                    # 0.0 to 1.0
    passed: bool                    # score >= threshold
    response_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    dimension: QualityDimension
    failure_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_cost_usd(self) -> float:
        input_cost = (self.input_tokens / 1_000_000) * 3.00
        output_cost = (self.output_tokens / 1_000_000) * 15.00
        return round(input_cost + output_cost, 6)


@dataclass
class EvalReport:
    """
    Aggregated results from running a full EvalSuite.
    This is what you show to stakeholders and store in CI.
    """
    suite_name: str
    results: list[EvalResult]
    duration_seconds: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.score for r in self.results)

    @property
    def average_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.latency_ms for r in self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.estimated_cost_usd for r in self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.results)

    def by_dimension(self) -> dict[str, list[EvalResult]]:
        """Group results by quality dimension."""
        groups: dict[str, list[EvalResult]] = {}
        for r in self.results:
            key = r.dimension.value
            groups.setdefault(key, []).append(r)
        return groups

    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    def print_report(self) -> None:
        """Print a formatted evaluation report."""
        print(f"\n{'═'*60}")
        print(f"EVAL REPORT: {self.suite_name}")
        print(f"{'═'*60}")
        print(f"Pass rate:        {self.pass_rate:.0%}  ({self.passed}/{self.total})")
        print(f"Average score:    {self.average_score:.2f}/1.00")
        print(f"Average latency:  {self.average_latency_ms:.0f}ms")
        print(f"Total tokens:     {self.total_tokens}")
        print(f"Total cost:       ${self.total_cost_usd:.4f}")
        print(f"Duration:         {self.duration_seconds:.1f}s")

        if self.failures():
            print(f"\n{'─'*60}")
            print(f"FAILURES ({self.failed}):")
            for r in self.failures():
                print(f"  ✗ [{r.case_id}] score={r.score:.2f}")
                for reason in r.failure_reasons:
                    print(f"      → {reason}")

        print(f"\nBY DIMENSION:")
        for dim, results in self.by_dimension().items():
            scores = [r.score for r in results]
            avg = statistics.mean(scores)
            passed = sum(1 for r in results if r.passed)
            print(f"  {dim:<15} {passed}/{len(results)} passed  avg={avg:.2f}")
        print(f"{'═'*60}\n")


# ─────────────────────────────────────────────
# 3. Scorers — functions that score AI output
# ─────────────────────────────────────────────

def score_keyword_presence(
    response_text: str,
    expected_keywords: list[str],
    forbidden_keywords: list[str],
) -> tuple[float, list[str]]:
    """
    Score based on keyword presence/absence.
    Returns (score 0.0-1.0, list of failure reasons).

    Scoring logic:
    - Start at 1.0
    - Deduct for each missing expected keyword
    - Deduct for each present forbidden keyword
    """
    if not expected_keywords and not forbidden_keywords:
        return 1.0, []

    failures = []
    text_lower = response_text.lower()
    total_checks = len(expected_keywords) + len(forbidden_keywords)
    penalties = 0

    for kw in expected_keywords:
        if kw.lower() not in text_lower:
            failures.append(f"Missing expected keyword: '{kw}'")
            penalties += 1

    for kw in forbidden_keywords:
        if kw.lower() in text_lower:
            failures.append(f"Found forbidden keyword: '{kw}'")
            penalties += 1

    score = max(0.0, 1.0 - (penalties / total_checks))
    return round(score, 2), failures


def score_format_compliance(
    response_text: str,
    expected_format: str | None,
) -> tuple[float, list[str]]:
    """Score based on output format compliance."""
    if not expected_format:
        return 1.0, []

    import json
    failures = []

    if expected_format == "json":
        try:
            json.loads(response_text.strip())
            return 1.0, []
        except json.JSONDecodeError as e:
            return 0.0, [f"Invalid JSON: {e.msg}"]

    if expected_format == "list":
        has_bullets = any(
            line.strip().startswith(("-", "*", "•", "·"))
            for line in response_text.split("\n")
        )
        has_numbers = any(
            line.strip()[:2] in [f"{i}." for i in range(1, 10)]
            for line in response_text.split("\n")
        )
        if has_bullets or has_numbers:
            return 1.0, []
        return 0.5, ["Response doesn't appear to be a list format"]

    if expected_format == "paragraph":
        word_count = len(response_text.split())
        if word_count >= 20:
            return 1.0, []
        return 0.5, [f"Response too short for paragraph: {word_count} words"]

    return 1.0, []


# ─────────────────────────────────────────────
# 4. The EvalSuite — runs cases and scores them
# ─────────────────────────────────────────────

class EvalSuite:
    """
    Runs a collection of EvalCases against a Claude client
    and produces a scored EvalReport.

    Usage:
        suite = EvalSuite("My Suite", client, pass_threshold=0.7)
        suite.add_case(EvalCase(...))
        report = suite.run()
        report.print_report()
    """

    def __init__(
        self,
        name: str,
        client: ClaudeClient,
        pass_threshold: float = 0.7,
    ):
        self.name = name
        self.client = client
        self.pass_threshold = pass_threshold
        self.cases: list[EvalCase] = []

    def add_case(self, case: EvalCase) -> "EvalSuite":
        """Add a case. Returns self for chaining."""
        self.cases.append(case)
        return self

    def add_cases(self, cases: list[EvalCase]) -> "EvalSuite":
        self.cases.extend(cases)
        return self

    def run(self, verbose: bool = False) -> EvalReport:
        """Run all cases and return a scored report."""
        start = time.monotonic()
        results = []

        for case in self.cases:
            if verbose:
                print(f"  Running: {case.id}...")
            result = self._run_case(case)
            results.append(result)
            if verbose:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {case.id}: score={result.score:.2f}")

        duration = time.monotonic() - start
        return EvalReport(
            suite_name=self.name,
            results=results,
            duration_seconds=duration,
        )

    def _run_case(self, case: EvalCase) -> EvalResult:
        """Run one case and score it."""
        try:
            response = self.client.chat(
                case.prompt,
                system=case.system_prompt,
            )

            # Score keyword presence
            kw_score, kw_failures = score_keyword_presence(
                response.text,
                case.expected_keywords,
                case.forbidden_keywords,
            )

            # Score format compliance
            fmt_score, fmt_failures = score_format_compliance(
                response.text,
                case.expected_format,
            )

            # Combined score (keywords weighted 70%, format 30%)
            if case.expected_format:
                score = (kw_score * 0.7) + (fmt_score * 0.3)
            else:
                score = kw_score

            all_failures = kw_failures + fmt_failures

            return EvalResult(
                case_id=case.id,
                score=round(score, 2),
                passed=score >= self.pass_threshold,
                response_text=response.text,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                dimension=case.dimension,
                failure_reasons=all_failures,
            )

        except Exception as exc:
            return EvalResult(
                case_id=case.id,
                score=0.0,
                passed=False,
                response_text="",
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                dimension=case.dimension,
                failure_reasons=[f"Exception: {exc}"],
            )