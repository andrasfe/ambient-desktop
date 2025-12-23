#!/bin/bash
# Start backend and frontend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load env
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "🚀 Starting Ambient Desktop..."

# Start PostgreSQL
docker compose up -d postgres 2>/dev/null || true
sleep 2

# Start backend
echo "Starting backend..."
cd "$PROJECT_ROOT/backend"
source venv/bin/activate
python run.py > /tmp/ambient-backend.log 2>&1 &
echo $! > /tmp/ambient-backend.pid
cd "$PROJECT_ROOT"

# Wait for backend
for i in {1..30}; do
    curl -s http://localhost:8000/ > /dev/null && break
    sleep 1
done

# Start frontend  
echo "Starting frontend..."
cd "$PROJECT_ROOT/frontend"
npm run dev > /tmp/ambient-frontend.log 2>&1 &
echo $! > /tmp/ambient-frontend.pid
cd "$PROJECT_ROOT"

sleep 3

echo ""
echo "✅ Running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo ""
echo "To stop: ./scripts/stop.sh"
