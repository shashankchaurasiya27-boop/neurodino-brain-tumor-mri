# ==============================================================================
# NeuroDINO® Diagnostic Platform - Hugging Face Spaces & Container Dockerfile
# Optimized for Fast Build, CPU PyTorch Inference, and Full Interactive UI
# ==============================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PROJECT_ROOT=/app \
    TORCH_HOME=/app/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Upgrade pip and install CPU-optimized PyTorch (fast build, no bloated CUDA wheels)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files and model assets with correct user ownership
COPY --chown=user:user . /app

# Ensure cache directories have correct permissions
RUN mkdir -p /app/.cache && chown -R user:user /app

# Switch to non-root user
USER user

# Hugging Face Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Launch the FastAPI app on port 7860
CMD ["python3", "-m", "uvicorn", "brain_tumor_frontend.app:app", "--host", "0.0.0.0", "--port", "7860"]

