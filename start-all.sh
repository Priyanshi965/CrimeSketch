#!/bin/bash

# CrimeSketch AI - Complete Startup Script
# Starts both the ML backend and web development server

set -e

PROJECT_DIR="/home/ubuntu/crimesketch_ai_web"
ML_BACKEND_PORT=8000
WEB_SERVER_PORT=3000

echo "=========================================="
echo "CrimeSketch AI - Startup Script"
echo "=========================================="
echo ""

# Check if embeddings have been generated
if [ ! -f "$PROJECT_DIR/ml_backend/embeddings/index.faiss" ]; then
    echo "⚠️  FAISS index not found!"
    echo "Running embedding generation first..."
    echo ""
    cd "$PROJECT_DIR"
    python3 ml_backend/scripts/generate_embeddings.py
    echo ""
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/.logs"

echo "Starting services..."
echo ""

# Start ML Backend in background
echo "[1/2] Starting ML Backend on port $ML_BACKEND_PORT..."
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
nohup python3 -m uvicorn server.ml_api:app --host 0.0.0.0 --port $ML_BACKEND_PORT > .logs/ml_backend.log 2>&1 &
ML_PID=$!
echo "✓ ML Backend started (PID: $ML_PID)"
echo "  Logs: .logs/ml_backend.log"

# Wait for ML backend to be ready
echo "Waiting for ML Backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:$ML_BACKEND_PORT/health > /dev/null 2>&1; then
        echo "✓ ML Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ ML Backend failed to start"
        exit 1
    fi
    sleep 1
done

echo ""

# Start Web Server in background
echo "[2/2] Starting Web Server on port $WEB_SERVER_PORT..."
cd "$PROJECT_DIR"
export ML_API_URL="http://localhost:$ML_BACKEND_PORT"
export NODE_ENV=development
nohup pnpm dev > .logs/web_server.log 2>&1 &
WEB_PID=$!
echo "✓ Web Server started (PID: $WEB_PID)"
echo "  Logs: .logs/web_server.log"
echo "  ML API URL: $ML_API_URL"

echo ""
echo "=========================================="
echo "✓ All services started successfully!"
echo "=========================================="
echo ""
echo "Access the application at:"
echo "  Web UI: http://localhost:$WEB_SERVER_PORT"
echo "  ML API: http://localhost:$ML_BACKEND_PORT"
echo "  API Docs: http://localhost:$ML_BACKEND_PORT/docs"
echo ""
echo "To stop all services, run:"
echo "  kill $ML_PID $WEB_PID"
echo ""
echo "View logs:"
echo "  ML Backend: tail -f .logs/ml_backend.log"
echo "  Web Server: tail -f .logs/web_server.log"
echo ""
