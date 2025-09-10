# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install specific packages needed for this MCP server
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
    PyPDF2

# Copy application files
COPY server_booking_mcp.py .
COPY hotel_local_rag.py .
COPY .env* ./

# Create directories for vector database and documents
RUN mkdir -p .hotel_vector_db /tmp/transformers_cache /tmp/huggingface

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Clean up to reduce image size
RUN pip cache purge && \
    rm -rf /root/.cache && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11 -name "__pycache__" -exec rm -rf {} + || true

# Expose port (if needed for debugging)
EXPOSE 8000

# Run the MCP server
CMD ["python", "server_booking_mcp.py"]