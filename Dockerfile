# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install only essential packages (NO EMBEDDING MODELS)
RUN pip install --no-cache-dir \
    fastmcp \
    gspread \
    google-auth \
    python-dotenv \
    fastapi \
    uvicorn \
    requests

# Copy application files
COPY . .

# Set environment variables
ENV RAILWAY_STARTUP_TIMEOUT=300
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DISABLE_EMBEDDING=1

# Clean up
RUN pip cache purge && \
    rm -rf /root/.cache && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]