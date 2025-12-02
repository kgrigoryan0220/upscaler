#!/bin/bash

# Start Real-ESRGAN API Server
# Usage: ./start_api.sh [port] [host]

PORT=${1:-8000}
HOST=${2:-0.0.0.0}

echo "Starting Real-ESRGAN API Server on ${HOST}:${PORT}"
echo "API Documentation will be available at http://${HOST}:${PORT}/docs"

python api_server.py

