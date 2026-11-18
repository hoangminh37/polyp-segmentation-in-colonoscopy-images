# Inference image: Gradio UI + REST API + Prometheus metrics on CPU.
#
#   python scripts/package_models.py      # writes checkpoints/ (models) and app/examples/
#   docker build -t polyp-seg .
#   docker run --rm -p 7860:7860 polyp-seg   # http://localhost:7860, API docs at /docs
#
# Only the release folder is copied in, never runs/ or the datasets (see .dockerignore).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code changes do not invalidate this layer.
ARG TORCH_VERSION=2.15.0
ARG TORCHVISION_VERSION=0.29.2
COPY requirements-serve.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
 && pip install -r requirements-serve.txt

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app src ./src
COPY --chown=app:app app ./app
COPY --chown=app:app checkpoints ./checkpoints

USER app
ENV HOST=0.0.0.0 PORT=7860 \
    POLYP_MODEL_DIR=/app/checkpoints WARMUP=all \
    NO_ALBUMENTATIONS_UPDATE=1 GRADIO_ANALYTICS_ENABLED=False MPLCONFIGDIR=/tmp
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["python", "-m", "app.server"]
