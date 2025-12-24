#!/bin/bash
# Setup script for Ambient Desktop Agent

set -e

echo "🚀 Setting up Ambient Desktop Agent..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Function to find Python 3.10+
find_python() {
    # Try python3.13, python3.12, python3.11, python3.10 in order
    for version in python3.13 python3.12 python3.11 python3.10; do
        if command -v "$version" >/dev/null 2>&1; then
            PYTHON_VERSION=$("$version" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
            PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
            if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
                echo "$version"
                return 0
            fi
        fi
    done
    
    # Try pyenv if available
    if command -v pyenv >/dev/null 2>&1; then
        echo "🐍 Found pyenv, checking for Python 3.10+..." >&2
        PYENV_VERSION=$(pyenv versions --bare 2>/dev/null | grep -E '^3\.(1[0-9]|[2-9][0-9])' | sort -V | tail -1)
        if [ -n "$PYENV_VERSION" ]; then
            echo "✅ Found Python $PYENV_VERSION via pyenv" >&2
            pyenv local "$PYENV_VERSION" 2>/dev/null || true
            if command -v python3 >/dev/null 2>&1; then
                PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
                PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
                PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
                if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
                    echo "python3"
                    return 0
                fi
            fi
        fi
    fi
    
    # Try Homebrew on macOS (check common paths directly, avoid slow brew commands)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🍺 Checking Homebrew for Python 3.10+..." >&2
        # Check common Homebrew Python locations without calling brew
        for prefix in "/opt/homebrew" "/usr/local"; do
            [ ! -d "$prefix" ] && continue
            for pyver in 3.13 3.12 3.11 3.10; do
                PYTHON_PATH="${prefix}/opt/python@${pyver}/bin/python3"
                [ ! -f "$PYTHON_PATH" ] && continue
                PYTHON_VERSION=$("$PYTHON_PATH" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
                [ -z "$PYTHON_VERSION" ] && continue
                PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
                PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
                if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ] 2>/dev/null; then
                    echo "$PYTHON_PATH"
                    return 0
                fi
            done
        done
    fi
    
    return 1
}

# Find Python 3.10+
echo "🔍 Looking for Python 3.10+..."
PYTHON_CMD=""
PYTHON_CMD=$(find_python) || true

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "❌ Python 3.10+ is required but not found."
    echo ""
    echo "Please install Python 3.10 or higher:"
    echo "  macOS: brew install python@3.11"
    echo "  Linux: Use your package manager (e.g., apt install python3.10)"
    echo "  Or use pyenv: pyenv install 3.11.0 && pyenv local 3.11.0"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
echo "✅ Found $PYTHON_VERSION"

# Check for required tools
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "⚠️  Docker is not installed (optional for some features)."; }

# Setup backend
echo "📦 Setting up backend..."
cd backend

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment with proper Python version
echo "🐍 Creating virtual environment with $PYTHON_VERSION..."
"$PYTHON_CMD" -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

cd "$PROJECT_ROOT"

# Setup frontend
echo "📦 Setting up frontend..."
cd frontend
npm install

cd "$PROJECT_ROOT"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENROUTER_API_KEY and COHERE_API_KEY"
echo "2. Run: ./scripts/start.sh"

