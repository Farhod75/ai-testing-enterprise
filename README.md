# Enterprise AI Testing Suite

A production-grade AI testing framework built with Python, pytest,
and the Anthropic Claude API — covering all 5 ISTQB testing levels.

## What This Project Tests
A FastAPI web application powered by Claude (Anthropic's AI),
tested across every level of the testing pyramid.

## Architecture

| Phase | Type | Tests | API Calls | Cost |
|-------|------|-------|-----------|------|
| 1 | Unit | 56 | None | $0.00 |
| 2 | Component | 25 | Mocked | $0.00 |
| 3 | Integration | 17 | Real | ~$0.01 |
| 4 | System/Eval | 19 | Real | ~$0.01 |
| 5 | E2E | 18 | Real | ~$0.01 |
| **Total** | **All levels** | **135** | | **~$0.03** |

## Key Features
- Reusable Claude API client with cost tracking
- Validator library for AI output quality testing
- Prompt pipeline framework with retry logic
- LLM evaluation harness scoring responses 0.0 to 1.0
- Safety testing — verifies Claude refuses harmful requests
- Playwright E2E tests against a real browser
- GitHub Actions CI/CD — all 5 phases run on every commit

## Tech Stack
- **Language:** Python 3.14
- **Testing:** pytest, pytest-asyncio, pytest-mock
- **AI:** Anthropic Claude API (claude-sonnet-4)
- **Web:** FastAPI, uvicorn
- **Browser:** Playwright (Chromium)
- **Validation:** Pydantic v2
- **CI/CD:** GitHub Actions

## Project Structure
```
ai-testing-enterprise/
├── src/
│   ├── ai_client/
│   │   ├── claude_client.py    # Reusable Claude wrapper
│   │   ├── validators.py       # AI output validators
│   │   ├── prompt_pipeline.py  # Prompt pipelines
│   │   └── eval_harness.py     # LLM eval framework
│   └── app.py                  # FastAPI web app
├── tests/
│   ├── unit/                   # Phase 1
│   ├── component/              # Phase 2
│   ├── integration/            # Phase 3
│   ├── system/                 # Phase 4
│   └── e2e/                    # Phase 5
├── conftest.py                 # Shared fixtures
├── pyproject.toml              # Project config
└── .github/workflows/          # CI/CD pipeline
```

## How to Run

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[test]"
playwright install chromium
```

### Add API Key
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run Each Phase
```bash
# Phase 1+2 — free, instant
pytest tests/unit/ tests/component/ -v

# Phase 3 — real API
pytest tests/integration/ -v -s

# Phase 4 — eval suite
pytest tests/system/ -v -s

# Phase 5 — E2E (start app first)
uvicorn src.app:app --port 8000
pytest tests/e2e/ -v -s

# All phases
pytest -v
```

## Certifications This Project Supports
- ISTQB CT-AI (Certified Tester AI Testing)
- ISTQB CT-GenAI (Certified Tester Generative AI Testing)
- Anthropic CCA Foundations

## Author
Farhod — Full Stack SDET