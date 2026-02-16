FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git sox build-essential \
        python3.10 python3.10-dev python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv

COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY api /app/api
COPY cosyvoice /app/cosyvoice
COPY fastcosyvoice /app/fastcosyvoice
COPY third_party/Matcha-TTS /app/third_party/Matcha-TTS

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python3.10", "-m", "api.run"]
