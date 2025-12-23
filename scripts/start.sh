#!/bin/bash
# Start the Ambient Desktop Agent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "🚀 Starting Ambient Desktop Agent..."

# Start PostgreSQL
echo "📦 Starting PostgreSQL..."
docker compose up -d postgres

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
until docker compose exec -T postgres pg_isready -U ambient -d ambient > /dev/null 2>&1; do
    sleep 1
done
echo "✅ PostgreSQL is ready"

# Start backend
echo "🔧 Starting backend..."
cd backend
source venv/bin/activate
python run.py &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for backend
echo "⏳ Waiting for backend..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Backend is ready"

# Start frontend
echo "🎨 Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

echo ""
echo "✅ Ambient Desktop Agent is running!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop..."

# Handle shutdown
trap "echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; docker compose down" EXIT

# Wait for processes
wait

