# Finance AI Engineer — Production Container
# Yahoo context: same pattern used for Qwen model serving

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Remove sensitive files
RUN rm -f .env

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["python3", "-m", "uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]