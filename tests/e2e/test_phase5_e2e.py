"""
tests/e2e/test_phase5_e2e.py

Phase 5: End-to-End Testing with Playwright

These tests control a REAL browser against a RUNNING web app.
They test what the user actually experiences.

BEFORE RUNNING:
1. Install Playwright browsers (one time only):
   playwright install chromium

2. Start the web app in a separate terminal:
   uvicorn src.app:app --port 8000

3. Run the tests:
   pytest tests/e2e/ -v -s

ISTQB CT-AI relevance:
- 3.6: End-to-end testing of AI systems
- 4.6: Testing AI user interfaces
- 5.6: Acceptance testing of AI features
"""

import os
import time
import pytest
import httpx

from playwright.sync_api import Page, expect, sync_playwright

# Base URL of the running app
BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def check_app_running():
    """
    Verify the app is running before any E2E test starts.
    Fails fast with a clear message if the server is not up.

    autouse=True means this runs automatically for every test
    in this file — no need to include it in each test.
    """
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        assert response.status_code == 200, (
            f"App returned {response.status_code} — expected 200"
        )
        data = response.json()
        print(f"\n✓ App is running — model: {data['model']}")
        print(f"✓ API key configured: {data['api_key_set']}")
    except httpx.ConnectError:
        pytest.skip(
            f"App not running at {BASE_URL}\n"
            f"Start it with: uvicorn src.app:app --port 8000"
        )


@pytest.fixture(scope="session")
def browser_context():
    """
    Single Playwright browser instance for all E2E tests.
    scope="session" — browser starts once, closes after all tests.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,   # True = no visible window (CI mode)
            # headless=False  # Uncomment to WATCH tests run in browser
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        yield context
        context.close()
        browser.close()


@pytest.fixture
def page(browser_context):
    """
    Fresh page for each test.
    scope="function" (default) — new tab per test, clean state.
    """
    page = browser_context.new_page()
    yield page
    page.close()


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — API Smoke Tests (no browser needed)
# ═══════════════════════════════════════════════════════════════

class TestAPISmoke:
    """
    Test the FastAPI endpoints directly with httpx.
    Fast — no browser startup needed.
    These run before the browser tests.
    """

    def test_health_endpoint_returns_200(self):
        """Health check must return 200 with correct shape."""
        response = httpx.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "api_key_set" in data

    def test_health_shows_api_key_configured(self):
        """API key should be configured in the running app."""
        response = httpx.get(f"{BASE_URL}/health")
        data = response.json()
        assert data["api_key_set"] is True, (
            "API key not set in the running app — "
            "ensure .env is loaded when starting uvicorn"
        )

    def test_chat_endpoint_accepts_valid_request(self):
        """POST /chat with valid prompt returns 200."""
        response = httpx.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Say exactly: OK"},
            timeout=30.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "input_tokens" in data
        assert "output_tokens" in data
        assert "model" in data

    def test_chat_response_contains_text(self):
        """Chat response text must not be empty."""
        response = httpx.post(
            f"{BASE_URL}/chat",
            json={"prompt": "Reply with one word: Hello"},
            timeout=30.0,
        )
        data = response.json()
        assert len(data["response"].strip()) > 0

    def test_chat_rejects_empty_prompt(self):
        """Empty prompt should return 422 (validation error)."""
        response = httpx.post(
            f"{BASE_URL}/chat",
            json={"prompt": ""},
            timeout=10.0,
        )
        assert response.status_code == 422

    def test_chat_with_system_prompt(self):
        """System prompt should influence the response."""
        response = httpx.post(
            f"{BASE_URL}/chat",
            json={
                "prompt": "Who are you?",
                "system": "You are TestBot. Always say 'I am TestBot' in your response."
            },
            timeout=30.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert "testbot" in data["response"].lower()

    def test_index_returns_html(self):
        """GET / should return the chat UI HTML page."""
        response = httpx.get(f"{BASE_URL}/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "prompt-input" in response.text
        assert "send-btn" in response.text


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — Browser UI Tests
# ═══════════════════════════════════════════════════════════════

class TestBrowserUI:
    """
    Browser-based UI tests using Playwright.
    These test what the user actually sees and interacts with.
    """

    def test_page_loads_successfully(self, page: Page):
        """The chat page should load with the correct title."""
        page.goto(BASE_URL)
        expect(page).to_have_title("AI Testing Enterprise — Chat")

    def test_input_field_is_visible(self, page: Page):
        """The prompt input field must be visible and enabled."""
        page.goto(BASE_URL)
        prompt_input = page.locator("#prompt-input")
        expect(prompt_input).to_be_visible()
        expect(prompt_input).to_be_enabled()

    def test_send_button_is_visible(self, page: Page):
        """The Send button must be visible and enabled on load."""
        page.goto(BASE_URL)
        send_btn = page.locator("#send-btn")
        expect(send_btn).to_be_visible()
        expect(send_btn).to_be_enabled()

    def test_initial_claude_greeting_is_displayed(self, page: Page):
        """Page should show Claude's greeting message on load."""
        page.goto(BASE_URL)
        chat_box = page.locator("#chat-box")
        expect(chat_box).to_contain_text("Hello")

    def test_user_can_type_in_input(self, page: Page):
        """User should be able to type in the prompt input."""
        page.goto(BASE_URL)
        prompt_input = page.locator("#prompt-input")
        prompt_input.fill("Hello Claude")
        expect(prompt_input).to_have_value("Hello Claude")

    def test_send_button_click_sends_message(self, page: Page):
        """Clicking Send should add user message to chat box."""
        page.goto(BASE_URL)
        prompt_input = page.locator("#prompt-input")
        send_btn = page.locator("#send-btn")

        prompt_input.fill("What is 2 + 2?")
        send_btn.click()

        # User message should appear immediately
        user_message = page.locator(".message.user")
        expect(user_message).to_be_visible()
        expect(user_message).to_contain_text("What is 2 + 2?")

    def test_enter_key_sends_message(self, page: Page):
        """Pressing Enter should send the message."""
        page.goto(BASE_URL)
        prompt_input = page.locator("#prompt-input")

        prompt_input.fill("Test message via Enter key")
        prompt_input.press("Enter")

        user_message = page.locator(".message.user")
        expect(user_message).to_be_visible()

    def test_input_clears_after_send(self, page: Page):
        """Input field should be empty after sending."""
        page.goto(BASE_URL)
        prompt_input = page.locator("#prompt-input")

        prompt_input.fill("Some message")
        prompt_input.press("Enter")

        # Input should clear immediately after send
        expect(prompt_input).to_have_value("")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — Full Journey E2E Tests (real API + browser)
# ═══════════════════════════════════════════════════════════════

class TestFullJourney:
    """
    Complete user journey tests.
    User opens page → types prompt → gets AI response.
    These are the most valuable and most expensive tests.
    Run last, run least frequently.
    """

    def test_complete_chat_journey(self, page: Page):
        """
        Full journey: load page → type → send → see response.
        This is the most important E2E test.
        """
        page.goto(BASE_URL)

        # Type a simple question
        page.locator("#prompt-input").fill("What is the capital of France? One word.")

        # Send it
        page.locator("#send-btn").click()
        # Wait for a NEW assistant message to appear after sending
        # Count existing messages first, then wait for one more
        page.wait_for_selector(".message.assistant:nth-child(n+2)", timeout=30_000) 

        # Get ALL assistant messages and check the last one
        assistant_messages = page.locator(".message.assistant")
        assistant_messages.last.wait_for(timeout=30_000)

        # Wait a moment for content to fully render
        page.wait_for_timeout(1000)

        # Verify response contains the correct answer
        response_text = assistant_messages.last.inner_text().lower()
        assert "paris" in response_text, (
            f"Expected 'paris' in response, got: {response_text}"
        )

    def test_send_button_disabled_while_waiting(self, page: Page):
        """Send button should be disabled while waiting for response."""
        page.goto(BASE_URL)

        page.locator("#prompt-input").fill("Tell me a very short story.")
        page.locator("#send-btn").click()

        # Button should be disabled immediately after click
        send_btn = page.locator("#send-btn")
        expect(send_btn).to_be_disabled()

        # Wait for response, button re-enables
        expect(send_btn).to_be_enabled(timeout=30_000)

    def test_multiple_messages_accumulate(self, page: Page):
        """Multiple messages should all appear in the chat box."""
        page.goto(BASE_URL)

        # Send first message
        page.locator("#prompt-input").fill("Say: First")
        page.locator("#send-btn").click()
        # Wait for first response
        expect(page.locator(".message.assistant").last).to_be_visible(timeout=30_000)

        # Send second message
        page.locator("#prompt-input").fill("Say: Second")
        page.locator("#send-btn").click()
        # Wait for second response
        page.wait_for_timeout(2000)

        # Both user messages should be in the chat
        user_messages = page.locator(".message.user")
        assert user_messages.count() >= 2