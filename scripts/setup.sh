#!/bin/bash
# Setup script for Ambient Desktop Agent

set -e

echo "🚀 Setting up Ambient Desktop Agent..."

# Check for required tools
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Setup backend
echo "📦 Setting up backend..."
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
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

