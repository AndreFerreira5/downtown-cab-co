FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY src/api /app/src/api

WORKDIR /app/src/api

RUN uv sync --no-dev

ENV PATH="/app/src/api/.venv/bin:$PATH"
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
