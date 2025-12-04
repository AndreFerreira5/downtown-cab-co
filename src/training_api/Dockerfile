FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/training_api /app/src/training_api

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "src.training_api.main"]