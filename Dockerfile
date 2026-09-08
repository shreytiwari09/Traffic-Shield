# One image, five services.
#
# All five FastAPI services import the same `services/` package and the same
# pinned requirements.txt — the only thing that differs between them is which
# ASGI app uvicorn is pointed at. Building five near-identical images would
# quintuple build time and disk for zero benefit, so compose builds this once
# and overrides `command:` per service. That mirrors how the code is actually
# organised: `services/shared/` is genuinely shared, not duplicated.
#
# Python 3.11 (not 3.12) deliberately: 3.11 is the interpreter this dependency
# set has been verified against locally, and chromadb 0.5.23 pins an older
# transitive stack that is fussier on newer interpreters.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before source: requirements.txt changes rarely, service code
# changes constantly, so this layer stays cached across normal edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only `services/` is copied. The services have no runtime import of
# data_pipeline (verified — it appears in their docstrings only), so the
# offline pipeline stays out of the image entirely.
COPY services/ ./services/

# Non-root. The one writable path a service needs is the bind-mounted
# chroma_data/ (SQLite requires write access for its journal even on
# read-only queries), so the UID must own that mount — see docker-compose.yml.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# 8000 app · 8001 orchestration · 8002 retrieval · 8003 llm · 8004 data
EXPOSE 8000 8001 8002 8003 8004

# Overridden per service in docker-compose.yml. Defaulting to the front door
# means `docker run` on this image alone still starts something meaningful.
CMD ["python", "-m", "uvicorn", "services.app_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
