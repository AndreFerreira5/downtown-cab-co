FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev


COPY src/inference_api /app/src/inference_api

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8080

CMD ["uvicorn", "src.inference_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
