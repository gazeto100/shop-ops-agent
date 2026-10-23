FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project || uv sync --no-dev --no-install-project
COPY src ./src
COPY data/products.sample.csv ./data/
RUN uv sync --no-dev
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "shop_ops_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
