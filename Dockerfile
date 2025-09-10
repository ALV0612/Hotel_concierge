# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional packages needed for the bot
RUN pip install --no-cache-dir \
    fastmcp \
    gspread \
    google-auth \
    sentence-transformers \
    transformers \
    torch \
    python-dotenv \
    langchain \
    langchain-community \
    chromadb \
    PyPDF2 \
    fastapi \
    uvicorn \
    requests

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p .hotel_vector_db /tmp/transformers_cache /tmp/huggingface

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface
ENV RAILWAY_STARTUP_TIMEOUT=300
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Clean up to reduce image size
RUN pip cache purge && \
    rm -rf /root/.cache && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11 -name "__pycache__" -exec rm -rf {} + || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the application
CMD ["python", "main.py"]