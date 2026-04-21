#!/bin/bash
# ── LingualSense — Start All Services ──────────────────────────────────────

echo ""
echo "  🔤 LingualSense — Indic Language Intelligence"
echo "  ─────────────────────────────────────────────"

# Kill anything already on these ports
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:7860 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null

# Start REST API
python3 -m uvicorn src.api.server:app --port 8000 > /tmp/api.log 2>&1 &
API_PID=$!

# Start Gradio dashboard
python3 scripts/dashboard.py --port 7860 > /tmp/gradio.log 2>&1 &
GRADIO_PID=$!

# Start HTML file server
python3 -m http.server 8080 --directory scripts > /tmp/html.log 2>&1 &
HTML_PID=$!

# Wait for services to come up
echo ""
echo "  Starting services..."
sleep 5

echo ""
echo "  ✅ All services running:"
echo ""
echo "  🌐 HTML Dashboard   →  http://localhost:8080/lingualsense_enhanced.html"
echo "  📊 Gradio Dashboard →  http://localhost:7860"
echo "  ⚡ REST API         →  http://localhost:8000"
echo "  📖 API Docs         →  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

# Keep script alive; kill children on exit
trap "kill $API_PID $GRADIO_PID $HTML_PID 2>/dev/null; echo '  Stopped.'; exit" INT TERM
wait
