#!/bin/bash
# Start backend and frontend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load env
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "🚀 Starting Ambient Desktop..."

# Start PostgreSQL
echo "Starting PostgreSQL..."
docker compose up -d postgres 2>/dev/null || true
# Wait for PostgreSQL to be ready (with timeout)
for i in {1..10}; do
    docker compose exec -T postgres pg_isready -U ambient >/dev/null 2>&1 && break
    sleep 0.5
done

# Start backend
echo "Starting backend..."
cd "$PROJECT_ROOT/backend"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./scripts/setup.sh first."
    exit 1
fi

# Check if port 8000 is already in use
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use. Stopping existing process..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Activate venv and start backend
source venv/bin/activate
python run.py > /tmp/ambient-backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/ambient-backend.pid
cd "$PROJECT_ROOT"

# Wait for backend with progress indicator
echo -n "Waiting for backend to start"
for i in {1..20}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo " ✅"
        break
    fi
    # Check if process died
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo ""
        echo "❌ Backend failed to start. Check /tmp/ambient-backend.log"
        tail -20 /tmp/ambient-backend.log
        exit 1
    fi
    echo -n "."
    sleep 0.5
done

# Final check
if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo ""
    echo "⚠️  Backend may still be starting. Check /tmp/ambient-backend.log"
fi

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
