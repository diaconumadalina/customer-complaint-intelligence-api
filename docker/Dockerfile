# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Avoid interactive prompts when installing system packages
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first, to use Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app ./app
COPY model ./model
COPY data ./data

# Expose the FastAPI default port
EXPOSE 8000

# Set environment variable for Python path
ENV PYTHONPATH=/app

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
