# Enterprise AI Testing Suite

[![CI](https://github.com/Farhod75/ai-testing-enterprise/actions/workflows/ai-tests.yml/badge.svg)](https://github.com/Farhod75/ai-testing-enterprise/actions/workflows/ai-tests.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-135-brightgreen)

A pytest framework for testing LLM-powered applications, built on the Anthropic Claude API.
It covers five test levels, from free unit tests to real-browser E2E, and includes a scored evaluation harness for non-deterministic AI output.

## What it demonstrates

- **Scored evals, not just pass/fail.** Each response gets a 0.0–1.0 score (keyword/forbidden-phrase checks 70%, format compliance 30%) and passes only above a configurable threshold (default 0.7).
- **Safety checks.** Verifies the model refuses harmful requests.
- **Consistency checks.** Repeated runs of the same prompt must stay within quality and length bounds.
- **Cost and latency tracking.** Every call reports tokens, latency and estimated USD; tests assert on budgets.
- **Cheap by design.** 81 of 135 tests run with no API calls. The full suite costs about $0.03.
- **CI on every commit.** GitHub Actions runs unit and component tests on every push and PR; real-API phases run on `main`.

## Test levels

| Phase | Level | Tests | API calls | Cost |
|-------|-------|-------|-----------|------|
| 1 | Unit | 56 | None | $0.00 |
| 2 | Component | 25 | Mocked | $0.00 |
| 3 | Integration | 17 | Real | ~$0.01 |
| 4 | System / Eval | 19 | Real | ~$0.01 |
| 5 | E2E (Playwright) | 18 | Real | ~$0.01 |
| | **Total** | **135** | | **~$0.03** |

## Architecture

```
src/
  ai_client/
    claude_client.py     Reusable Claude wrapper with cost tracking
    validators.py        Output validators (length, keywords, JSON, patterns)
    prompt_pipeline.py   Prompt pipelines with retry logic
    eval_harness.py      EvalCase / EvalSuite / EvalReport, scoring, thresholds
  app.py                 FastAPI chat app under test
tests/
  unit/  component/  integration/  system/  e2e/
conftest.py              Shared fixtures
```

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[test]"
playwright install chromium
copy .env.example .env      # then add your ANTHROPIC_API_KEY
```

```powershell
# Free and instant (Phases 1-2)
pytest tests/unit tests/component -v

# Real API (Phases 3-4)
pytest tests/integration tests/system -v -s

# E2E (Phase 5): start the app, then run the tests
uvicorn src.app:app --port 8000
pytest tests/e2e -v -s

# Everything
pytest -v
```

## Design notes and limitations

- **The keyword oracle is a heuristic.** It is deterministic and cheap, but can mis-score a correct answer phrased differently. I hit this in CI (a safety case was flaky until I refined its oracle). A model-as-judge scorer is the next step.
- **Real-API phases depend on model behavior,** so they can drift between model versions. Thresholds are set to catch regressions, not to demand identical output.
- **Single model, single app.** The suite shows the testing approach; it is not a benchmark across models.

## Tech stack

Python 3.11+, pytest, pytest-asyncio, pytest-mock, respx, Pydantic v2, FastAPI, Playwright (Chromium), Anthropic Claude API, GitHub Actions.

## Context

Built while preparing for the ISTQB CT-AI and CT-GenAI certifications (both earned) to apply their test-design ideas to a working system.

## Author

Farhod Elbekov, SDET / AI QA Engineer. [GitHub](https://github.com/Farhod75)
