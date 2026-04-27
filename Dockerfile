FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p models data/raw data/policies logs

# Pre-build: generate data, train model, build RAG index
RUN python -m scripts.train_model --generate --samples 20000

# Hugging Face Spaces uses 7860. Render/Railway use $PORT.
EXPOSE 7860

ENV PORT=7860
ENV APP_ENV=production
ENV LOG_LEVEL=INFO

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1
