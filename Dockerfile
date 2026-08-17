# No CUDA base here. The kernel module isn't in this tree, so a CUDA image would
# only be misleading — this builds a container that runs the synthetic backend.
# A real image needs backend_cuda.py and a matching nvidia/cuda base.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GROKBOT_LOG_FORMAT=json

WORKDIR /app

# Dependency layer first so source edits don't invalidate it. There are no
# runtime deps, so this is nearly a no-op — kept for when that changes.
COPY pyproject.toml README.md LICENSE NOTICE ./
RUN pip install --upgrade pip && pip install -e . || true

COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install -e .

# Don't run as root. The tool sandbox forks; a compromised tool inheriting root
# in the container is the difference between a contained failure and a bad day.
RUN useradd --create-home --uid 10001 grokbot \
    && chown -R grokbot:grokbot /app
USER grokbot

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"

ENTRYPOINT ["python", "-m", "grokbot"]
CMD ["serve", "--config", "configs/serving.yaml", "--host", "0.0.0.0", "--port", "8080"]
