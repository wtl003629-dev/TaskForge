FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY backend ./backend
RUN python -m pip install --upgrade pip && python -m pip install .

COPY AGENTS.md ./AGENTS.md
COPY docs ./docs
COPY config ./config
COPY migrations ./migrations
COPY scripts ./scripts
COPY frontend/package.json ./frontend/package.json
COPY frontend/src ./frontend/src

RUN groupadd --gid 10001 taskforge \
    && useradd --uid 10001 --gid taskforge --no-create-home taskforge \
    && mkdir -p /app/.taskforge \
    && chown -R taskforge:taskforge /app/.taskforge

USER taskforge
EXPOSE 8000

CMD ["uvicorn", "taskforge.app:app", "--host", "0.0.0.0", "--port", "8000"]
