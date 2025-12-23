#!/bin/bash
# Launch Chrome with remote debugging for browser takeover

PORT=9222

# Check if already running
if curl -s "http://localhost:$PORT/json/version" > /dev/null 2>&1; then
    echo "✅ Chrome already running on port $PORT"
    curl -s "http://localhost:$PORT/json/version" | grep -E '"Browser"|"webSocketDebuggerUrl"'
    exit 0
fi

echo "🌐 Starting Chrome with remote debugging on port $PORT..."

# Detect OS and launch Chrome
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CHROME_CMD=$(which google-chrome || which chromium || which chromium-browser)
    if [ -z "$CHROME_CMD" ]; then
        echo "❌ Chrome/Chromium not found"
        exit 1
    fi
    "$CHROME_CMD" \
        --remote-debugging-port=$PORT \
        --user-data-dir="$HOME/.ambient-chrome-profile" \
        &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
        --remote-debugging-port=$PORT \
        --user-data-dir="$HOME/.ambient-chrome-profile" \
        &
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

# Wait for Chrome to start
echo "Waiting for Chrome..."
for i in {1..10}; do
    if curl -s "http://localhost:$PORT/json/version" > /dev/null 2>&1; then
        echo ""
        echo "✅ Chrome ready!"
        echo ""
        curl -s "http://localhost:$PORT/json/version" | grep -E '"Browser"|"webSocketDebuggerUrl"'
        echo ""
        echo "Add to .env: BROWSER_CDP_URL=http://localhost:$PORT"
        exit 0
    fi
    sleep 1
done

echo "❌ Chrome failed to start"
exit 1

