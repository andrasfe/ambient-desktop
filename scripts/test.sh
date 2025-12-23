#!/bin/bash
# Run tests for Ambient Desktop Agent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "🧪 Running tests..."

# Backend tests
echo "📦 Running backend tests..."
cd backend
source venv/bin/activate
pip install aiosqlite  # Required for SQLite async in tests
pytest tests/ -v --tb=short

echo ""
echo "✅ All tests passed!"

