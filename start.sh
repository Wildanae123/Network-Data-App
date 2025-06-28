#!/bin/bash
echo "Starting application on port ${PORT:-8080}"
exec gunicorn backend.production_server:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers ${GUNICORN_WORKERS:-1} \
    --worker-class sync \
    --timeout ${GUNICORN_TIMEOUT:-60} \
    --keep-alive ${GUNICORN_KEEPALIVE:-2} \
    --max-requests ${MAX_REQUESTS:-100} \
    --max-requests-jitter 10 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --capture-output