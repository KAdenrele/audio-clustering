FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# 1. Install system dependencies
# Added libsndfile1 (required by soundfile) and ffmpeg (for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# 3. Install dependencies
# Corrected: uv automatically detects pyproject.toml in the current directory for project dependencies.
RUN uv pip install --system pyproject.toml


COPY . .

CMD ["uv", "run", "scripts/ProcessingCluster.py"]