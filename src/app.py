"""
src/app.py — FastAPI web application powered by Claude.

This is the system under test for Phase 5 E2E testing.
A minimal but real AI-powered chat interface.

Run locally:
    uvicorn src.app:app --reload --port 8000

Then open: http://localhost:8000
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import anthropic
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    system: str | None = None


class ChatResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    api_key_set: bool


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — /chat endpoint will fail")
    yield
    logger.info("App shutting down")


app = FastAPI(
    title="AI Testing Enterprise — Chat API",
    description="Claude-powered chat API for E2E testing",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — tested first in E2E smoke tests."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    return HealthResponse(
        status="ok",
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        api_key_set=bool(api_key),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — sends prompt to Claude, returns response.
    This is the core endpoint tested in Phase 5 E2E tests.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured",
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {
            "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            "max_tokens": 512,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system

        message = client.messages.create(**kwargs)

        return ChatResponse(
            response=message.content[0].text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=message.model,
        )

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Simple chat UI — tested with Playwright in Phase 5.
    Pure HTML/JS — no framework needed for testing purposes.
    """
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Testing Enterprise — Chat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, sans-serif;
            background: #0f1117;
            color: #e2e8f0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .container {
            width: 100%;
            max-width: 720px;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        h1 {
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            color: #a78bfa;
        }
        #chat-box {
            background: #1e2130;
            border: 1px solid #2d3148;
            border-radius: 12px;
            padding: 1.5rem;
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .message.user {
            background: #2d1f4e;
            border-left: 3px solid #a78bfa;
            align-self: flex-end;
            max-width: 85%;
        }
        .message.assistant {
            background: #1a2744;
            border-left: 3px solid #60a5fa;
            align-self: flex-start;
            max-width: 90%;
        }
        .message .label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
            opacity: 0.7;
        }
        .input-row {
            display: flex;
            gap: 0.75rem;
        }
        #prompt-input {
            flex: 1;
            background: #1e2130;
            border: 1px solid #2d3148;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: #e2e8f0;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        #prompt-input:focus {
            border-color: #a78bfa;
        }
        #send-btn {
            background: #7c3aed;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            white-space: nowrap;
        }
        #send-btn:hover { background: #6d28d9; }
        #send-btn:disabled {
            background: #4c1d95;
            cursor: not-allowed;
            opacity: 0.6;
        }
        #status {
            font-size: 0.8rem;
            text-align: center;
            min-height: 1.2rem;
            color: #94a3b8;
        }
        #status.error { color: #f87171; }
        #status.loading { color: #fbbf24; }
        .token-info {
            font-size: 0.75rem;
            color: #64748b;
            text-align: right;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Testing Enterprise</h1>
        <div id="chat-box">
            <div class="message assistant">
                <div class="label">Claude</div>
                <div>Hello! I'm Claude. Ask me anything.</div>
            </div>
        </div>
        <div class="input-row">
            <input
                type="text"
                id="prompt-input"
                placeholder="Type your message..."
                autocomplete="off"
            />
            <button id="send-btn">Send</button>
        </div>
        <div id="status"></div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        const status = document.getElementById('status');

        function addMessage(role, text, tokens) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `
                <div class="label">${role === 'user' ? 'You' : 'Claude'}</div>
                <div class="content">${text}</div>
                ${tokens ? `<div class="token-info">${tokens} tokens</div>` : ''}
            `;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function sendMessage() {
            const prompt = promptInput.value.trim();
            if (!prompt) return;

            addMessage('user', prompt);
            promptInput.value = '';
            sendBtn.disabled = true;
            status.textContent = 'Thinking...';
            status.className = 'loading';

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt }),
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Request failed');
                }

                const data = await res.json();
                addMessage(
                    'assistant',
                    data.response,
                    `${data.input_tokens + data.output_tokens}`
                );
                status.textContent = '';
                status.className = '';

            } catch (err) {
                status.textContent = `Error: ${err.message}`;
                status.className = 'error';
            } finally {
                sendBtn.disabled = false;
                promptInput.focus();
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        promptInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
    """)