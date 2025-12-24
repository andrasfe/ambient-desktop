# Ambient Desktop Agent

Control your computer through natural language. Browse the web, manage files, automate tasks.

## Quick Start

```bash
# 1. Setup (first time only)
./scripts/setup.sh

# 2. Edit .env with your API keys
cp env.example .env
nano .env

# 3. Start everything
./scripts/start.sh

# 4. Open http://localhost:3000
```

## Browser Takeover

To let the agent use your logged-in browser sessions (LinkedIn, Gmail, etc):

```bash
# Terminal 1: Start Chrome with remote debugging
./scripts/chrome.sh

# Terminal 2: Start the agent
./scripts/start.sh
```

The agent will control your existing browser with all your logins intact.

> **Note**: When using browser takeover, open the frontend (localhost:3000) in **Firefox or Safari** instead of Chrome. The agent's browser control affects all Chrome tabs, which can cause UI issues if the frontend is also in Chrome.

## Scripts

| Script | Description |
|--------|-------------|
| `./scripts/setup.sh` | Install dependencies (run once) |
| `./scripts/start.sh` | Start backend + frontend |
| `./scripts/stop.sh` | Stop everything |
| `./scripts/chrome.sh` | Launch Chrome for browser takeover |
| `./scripts/test.sh` | Run tests |

## Configuration

Key settings in `.env`:

```env
# LLM (OpenRouter or local Ollama)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Browser takeover (optional)
BROWSER_CDP_URL=http://localhost:9222
```

See `env.example` for all options.

## Tech Stack

- **Backend**: FastAPI, LangGraph, Playwright, PostgreSQL
- **Frontend**: React, Vite, Tailwind, Zustand
- **LLM**: OpenRouter or local Ollama

## License

MIT
