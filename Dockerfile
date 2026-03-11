FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

CMD ["sh", "-c", "mkdir -p data/annotation data/pipeline data/pipeline/prompts && touch data/annotation/annotations.jsonl data/annotation/comments.jsonl && uv run python -m annotation.dashboard"]
