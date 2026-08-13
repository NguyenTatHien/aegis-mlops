# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# CPU-only torch wheel is the point of this stage: download.pytorch.org/whl/cpu
# serves plain "torch" without the ~2GB CUDA runtime bundled in the PyPI build.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

FROM python:3.11-slim AS runtime

RUN groupadd --system aegis && useradd --system --gid aegis --home-dir /app --create-home aegis

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ /app/src/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN chown -R aegis:aegis /app
USER aegis

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "aegis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
