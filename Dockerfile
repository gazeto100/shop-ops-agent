FROM python:3.12.15-slim-trixie AS base
# UV_LINK_MODE=copy: the cache is a separate mount, so hardlinks into it can't work.
# UV_PYTHON_DOWNLOADS=never: use the image's Python 3.12, never fetch another one.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Pinned to the uv that wrote uv.lock; bump together with the lock.
RUN pip install --no-cache-dir uv==0.13.5
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Cache mount keeps uv's download cache out of the image layers.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY data/products.sample.csv ./data/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
# Unprivileged runtime user; the home dir is for caches (e.g. Hugging Face models, #3).
RUN useradd --create-home --uid 10001 app
USER app
# Run the venv directly: `uv run` would re-sync the project on every container start.
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "shop_ops_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
