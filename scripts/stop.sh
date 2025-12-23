#!/bin/bash
# Stop backend and frontend

echo "🛑 Stopping Ambient Desktop..."

# Kill backend
if [ -f /tmp/ambient-backend.pid ]; then
    kill $(cat /tmp/ambient-backend.pid) 2>/dev/null
    rm /tmp/ambient-backend.pid
fi
pkill -f "python run.py" 2>/dev/null || true

# Kill frontend
if [ -f /tmp/ambient-frontend.pid ]; then
    kill $(cat /tmp/ambient-frontend.pid) 2>/dev/null
    rm /tmp/ambient-frontend.pid
fi
pkill -f "vite" 2>/dev/null || true

# Stop PostgreSQL
docker compose down 2>/dev/null || true

echo "✅ Stopped"

