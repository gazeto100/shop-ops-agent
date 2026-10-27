.PHONY: setup lint test test-integration evals ingest serve docker

setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

# Tool logic must stay at 100% line+branch coverage (#6).
test:
	uv run pytest tests/unit -q --cov=shop_ops_agent.agent.tools --cov-branch \
		--cov-report=term-missing:skip-covered --cov-fail-under=100

test-integration:
	uv run pytest tests/integration -q

evals:
	uv run pytest tests/evals -q

ingest:
	uv run python -m shop_ops_agent.ingest data/products.sample.csv

serve:
	uv run uvicorn shop_ops_agent.api.app:app --reload --port 8000

docker:
	docker build -t shop-ops-agent .
