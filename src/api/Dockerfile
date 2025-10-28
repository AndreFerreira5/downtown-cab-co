FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY src/api /app/src/api

RUN uv sync

CMD ["uv", "run", "src/api/app.py"]
