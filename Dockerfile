# Multi-stage: the build stage carries pip and wheels, the runtime stage does
# not. Keeps the shipped image smaller and removes tooling an attacker could
# use if they got a shell.
FROM python:3.14-slim AS build

WORKDIR /build
COPY requirements.txt .
# --no-cache-dir so pip's cache never lands in a layer.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.14-slim AS runtime

# Run as a non-root user. Nothing here needs privileges, and a container that
# does not need root should not have it.
RUN useradd --create-home --uid 10001 hlr
WORKDIR /app

COPY --from=build /install /usr/local
COPY --chown=hlr:hlr app/ ./app/
COPY --chown=hlr:hlr static/ ./static/
COPY --chown=hlr:hlr mock_frontier/ ./mock_frontier/
# The fitted calibrators, but not the rest of eval/ — the harness needs
# scikit-learn and never runs in the serving image. Without this file
# CONFIDENCE_MODE=calibrated refuses to start (by design), so it ships even
# though the default mode ignores it.
COPY --chown=hlr:hlr eval/calibrator.json ./eval/calibrator.json

USER hlr

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json

EXPOSE 8000

# Liveness only — a healthcheck that tests Redis would mark a *healthy*
# gateway unhealthy during a Redis blip, turning a degradation into a restart
# loop. Readiness belongs to the orchestrator, via /readyz.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/livez', timeout=2).status == 200 else 1)"

# Ollama is NOT in this image. It needs the host's GPU/Metal and its own model
# storage, so it stays a host service and the container reaches it via
# OLLAMA_BASE_URL. Bundling it would mean CPU-only inference in a container.
#
# Worker count is deliberately left to the deployer: each worker holds its own
# numpy replica and HTTP pool, so the right number depends on available RAM
# (2-4 on an 8 GB host).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
