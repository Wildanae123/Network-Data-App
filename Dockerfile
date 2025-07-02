# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NODE_ENV=production \
    FLASK_ENV=production \
    PYTHONPATH=/app

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements*.txt ./

# Install Python dependencies
RUN echo "Installing production dependencies..." && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy frontend package files and install Node dependencies
COPY frontend/package*.json ./frontend/
RUN cd frontend && \
    echo "Installing frontend dependencies..." && \
    npm ci --production=false --silent

# Copy all source code
COPY . .

# Build frontend
RUN cd frontend && \
    echo "Building frontend..." && \
    npm run build && \
    echo "Frontend build completed" && \
    ls -la dist/

# Create necessary directories
RUN mkdir -p uploads output logs && \
    echo "Created necessary directories"

# Verify backend can import
RUN cd backend && \
    python -c "import production_server; print('✅ Backend imports successfully')" && \
    echo "Backend verification completed"

# Set proper permissions
RUN chmod -R 755 backend/ && \
    chmod 644 backend/production_server.py

# Copy and set up startup script
COPY required/start.sh ./
RUN chmod +x start.sh

# Use startup script
CMD ["./start.sh"]