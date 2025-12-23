# Ambient Desktop Agent

An always-on multi-agent system for computer automation. Control your computer through natural language - browse the web, manage files, and automate tasks.

## Features

- **Chat Interface**: Natural language instructions to control your computer
- **Browser Automation**: Navigate, click, type, and extract data from web pages via Playwright
- **File Operations**: Read, write, and manage local files
- **Multi-Agent System**: Spawn multiple agents for parallel task execution
- **Real-time Updates**: Live activity logs and agent status via WebSocket
- **Task Queue**: Scheduled and priority-based task execution
- **MCP Integration**: Extend capabilities via Model Context Protocol

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│  │   Chat   │  │ Activity Log │  │    Agent Status     │   │
│  └──────────┘  └─────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │ WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐    │
│  │ Coordinator  │  │ Scheduler │  │   MCP Client     │    │
│  │    Agent     │  │           │  │                  │    │
│  └──────┬───────┘  └─────┬─────┘  └──────────────────┘    │
│         │                │                                  │
│  ┌──────▼───────────────▼──────────────────────────────┐  │
│  │              Worker Agents                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │  Browser   │  │    File    │  │   Custom   │    │  │
│  │  │  (Playwright)│  │            │  │            │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PostgreSQL  │  │  OpenRouter  │  │    Cohere    │     │
│  │  (Task Queue)│  │   (LLM API)  │  │ (Embeddings) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Privacy & Security

This agent can run **100% locally** with no data sent to cloud APIs:

### Recommended: Local LLM with Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose based on your hardware)
ollama pull llama3.1:8b      # Good balance (needs ~8GB RAM)
ollama pull phi3:mini        # Lightweight (needs ~4GB RAM)
ollama pull deepseek-coder-v2:16b  # Best for coding tasks
```

Configure `.env`:
```env
OPENROUTER_API_KEY=ollama
OPENROUTER_BASE_URL=http://localhost:11434/v1
OPENROUTER_MODEL=llama3.1:8b
```

### If Using Cloud APIs

Enable privacy mode to minimize data exposure:
```env
PRIVACY_MODE=true  # Only sends task descriptions, not page content
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for PostgreSQL)
- **Ollama** (recommended for privacy) - or OpenRouter API key

### Setup

1. Clone and setup:
```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

2. Configure your API keys in `.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
COHERE_API_KEY=your-cohere-key-here
```

3. Start the system:
```bash
./scripts/start.sh
```

4. Open http://localhost:3000 in your browser

## Configuration

Copy `env.example` to `.env` and configure:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM |
| `OPENROUTER_MODEL` | LLM model (default: anthropic/claude-3.5-sonnet) |
| `COHERE_API_KEY` | Cohere API key for embeddings |
| `SCHEDULER_INTERVAL_SECONDS` | Task polling interval (default: 30) |
| `MAX_CONCURRENT_AGENTS` | Max parallel agents (default: 5) |
| `MCP_SERVERS` | JSON array of MCP server configs |
| `BROWSER_CDP_URL` | Connect to existing browser (e.g., `http://localhost:9222`) |
| `BROWSER_USER_DATA_DIR` | Directory for persistent browser profile |
| `BROWSER_HEADLESS` | Run browser headless (default: true) |

## Taking Over Your Browser Session

The agent can take over your existing browser where you're already logged in (e.g., LinkedIn, Gmail):

### Method 1: CDP Connection (Recommended)

1. Start Chrome with remote debugging:
```bash
# Linux
google-chrome --remote-debugging-port=9222

# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Or use Chromium
chromium --remote-debugging-port=9222
```

2. Add to your `.env`:
```env
BROWSER_CDP_URL=http://localhost:9222
```

3. The agent will now control YOUR browser with all your logins intact!

### Method 2: Persistent Profile

The agent maintains its own profile that persists cookies between sessions:

```env
BROWSER_USER_DATA_DIR=~/.ambient-browser-profile
```

Log in once through the agent, and it stays logged in.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | System status |
| `GET /health` | Health check |
| `WS /chat/ws` | WebSocket chat |
| `POST /chat/message` | Send chat message |
| `GET /tasks/` | List tasks |
| `POST /tasks/` | Create task |
| `GET /agents/` | List agents |
| `GET /agents/logs/` | Activity logs |

## Development

### Running Tests

```bash
./scripts/test.sh
```

### Backend Only

```bash
cd backend
source venv/bin/activate
python run.py
```

### Frontend Only

```bash
cd frontend
npm run dev
```

## MCP Integration

Add MCP servers in your `.env`:

```env
MCP_SERVERS=[{"name": "filesystem", "command": "mcp-server-filesystem", "args": ["/home"]}]
```

## Technology Stack

**Backend**
- FastAPI + Uvicorn
- SQLAlchemy + asyncpg
- Playwright for browser automation
- APScheduler for task scheduling
- httpx for async HTTP

**Frontend**
- React 18 + Vite
- Zustand for state management
- Tailwind CSS for styling
- lucide-react for icons

**Infrastructure**
- PostgreSQL for persistence
- Docker Compose for services
- OpenRouter for LLM
- Cohere for embeddings

## License

MIT

