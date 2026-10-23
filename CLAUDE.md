# shop-ops-agent - Claude Code project instructions

## What this project is
An AI operations agent for a small e-commerce store (electro.marketpage.bg).
It answers customer/operator questions over the product catalog using RAG, and
performs actions through tools (stock/price lookup, supplier price comparison,
order drafts). Exposed as a FastAPI service. Built as a portfolio project that
must demonstrate **production-grade AI engineering practices**, not just a demo.

The owner is an experienced C++/PHP engineer who is new to Python. Write
idiomatic, modern Python and explain non-obvious Python idioms briefly in PR
descriptions or comments when you introduce them (e.g. `Protocol`, context
managers, `async`), but do not over-comment ordinary code.

## Non-negotiables (quality bar)
- **Every feature ships with tests.** No PR without `pytest` coverage for the new code.
- **Evals are first-class.** Any change to prompts, retrieval, or tools must run
  `make evals` and must not regress the golden set (see `tests/evals/`).
- **Type hints everywhere.** `mypy --strict` passes on `src/`.
- **No secrets in code.** All keys come from `.env` via `src/shop_ops_agent/config.py`.
- **Deterministic tests.** Unit tests never call a real LLM. Use the `FakeLLM`
  in `tests/conftest.py`. Only `tests/evals/` may call a real provider, and only
  when `RUN_LIVE_EVALS=1`.
- **Small, reviewable commits.** One issue → one branch → one PR. Reference the
  issue number in the commit message (`feat(rag): hybrid retrieval (#4)`).

## Stack (fixed - do not swap without asking)
- Python 3.12, `uv` for env/deps, `ruff` (lint+format), `mypy`, `pytest`
- FastAPI + Uvicorn, Pydantic v2 for all request/response/tool schemas
- LLM: provider-agnostic via `src/shop_ops_agent/llm.py`
  (`openai`, `anthropic`, or `ollama` selected by `LLM_PROVIDER` env var)
- Embeddings: `sentence-transformers` (local, `intfloat/multilingual-e5-small`)
  - Bulgarian + English product text, no API cost for indexing
- Vector store: ChromaDB (persistent, `./data/chroma`)
- Tracing: Langfuse (optional, enabled when `LANGFUSE_*` keys exist)
- Container: single `Dockerfile`, CI via GitHub Actions
- Installed SDK versions are newer than any model's training data (`openai` 3.x, `anthropic` 2.x,
  `chromadb` 1.5, `sentence-transformers` 6.x). Before writing code against them, read the
  installed package's docstrings or `uv run python -c "import X; help(X.Client)"` - do not
  write API calls from memory.

Do NOT introduce LangChain/LlamaIndex. The point of this project is to show
understanding of the primitives (retrieval, tool loop, evals), not framework use.

## Layout
```
src/shop_ops_agent/
  config.py        # pydantic-settings; the only place env vars are read
  llm.py           # provider abstraction: complete(messages, tools) -> Response
  ingest/          # CSV/feed -> normalized Product -> chunks -> embeddings
  rag/             # retriever (vector + BM25 hybrid), reranker, citation builder
  agent/           # tool definitions (pydantic), tool registry, agent loop, memory
  api/             # FastAPI app, routers, schemas
  evals/           # eval runner, judges, metrics
tests/
  unit/            # fast, no network
  evals/           # golden set (JSONL) + eval tests
data/
  products.sample.csv   # committed sample; real feed is gitignored
docs/
  ARCHITECTURE.md, DECISIONS.md (ADR-style, append-only)
```

## Commands
```
make setup      # uv sync + pre-commit install
make lint       # ruff check + ruff format --check + mypy
make test       # pytest tests/unit
make evals      # pytest tests/evals (needs RUN_LIVE_EVALS=1 + provider key)
make ingest     # python -m shop_ops_agent.ingest data/products.sample.csv
make serve      # uvicorn shop_ops_agent.api.app:app --reload
make docker     # docker build -t shop-ops-agent .
```
Run `make lint test` before every commit. CI runs the same.

## Working agreements for Claude Code
1. Before starting an issue, read it (`gh issue view N`) and restate the
   acceptance criteria in your plan. If criteria are ambiguous, ask - do not guess.
2. Prefer the smallest change that satisfies the acceptance criteria. Do not
   add features, abstractions, or "while I'm here" refactors.
3. When you add a dependency, justify it in one line in `docs/DECISIONS.md`.
4. When you change a prompt, add/adjust at least one golden-set case that
   would have caught a regression.
5. After finishing an issue: run `make lint test`, update `README.md` if the
   user-facing behaviour changed, then open a PR with `gh pr create` using the
   template in `.github/PULL_REQUEST_TEMPLATE.md`.
6. Never commit `.env`, `data/chroma/`, or real product feeds.

## Domain notes
- Products have: `sku`, `name` (BG), `category`, `price_bgn`, `price_eur`,
  `stock_qty`, `supplier_sku`, `supplier_price_bgn`, `description`, `warranty_months`.
- Currency: Bulgaria is in the eurozone since 2026; show EUR primary, BGN secondary.
- Customer-facing answers are in Bulgarian unless the user writes in English.
- The agent must never invent prices or stock. If a tool returns nothing, say so.
