FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Generate sample if not already present, then start the dashboard
CMD ["sh", "-c", "uv run python -m annotation.generate_sample && uv run python -m annotation.dashboard"]
