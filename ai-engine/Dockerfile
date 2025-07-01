FROM python:3.11-slim

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies with timeout and retry settings
RUN pip install --no-cache-dir --upgrade pip --timeout 1000 --retries 3 && \
    pip install --no-cache-dir --timeout 1000 --retries 3 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs && \
    chmod 755 models logs

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn with production configuration
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
