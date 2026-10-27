# shop-ops-agent

[![CI](https://github.com/gazeto100/shop-ops-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gazeto100/shop-ops-agent/actions/workflows/ci.yml)

AI operations agent for a small e-commerce store: RAG over the product catalog,
tool-calling for stock/price/supplier lookups, FastAPI service, eval suite, tracing, CI.

> Status: scaffolding. See `docs/ISSUES.md` for the build plan.

## Why
Most "chat with your data" demos stop at retrieval. This project shows the parts that
matter in production: provider-agnostic LLM layer, deterministic tests, an eval suite
that gates CI, observability, and a container you can actually deploy.

## Architecture
```
CSV / supplier feed ──► ingest ──► chunks ──► embeddings ──► ChromaDB
                                                                │
client ──► FastAPI /chat ──► agent loop ──► retriever (hybrid) ◄┘
                               │  └──► tools: stock, price, supplier_compare, order_draft
                               └──► LLM (openai | anthropic | ollama)
                                      │
                                   Langfuse traces
```

## Quick start
```bash
cp .env.example .env          # fill LLM_PROVIDER + key
make setup
make ingest
make serve
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo","message":"Има ли наличен бойлер 80 литра и каква е цената?"}'
```

`/chat` returns `{"answer": "...", "citations": [{"sku": "...", "name": "..."}], "tool_calls": []}`.
The answer is grounded in the retrieved catalog excerpt and written in the user's language
(Bulgarian unless the message is in English). `citations` lists only retrieved products the
answer actually cites. The assistant never states prices or stock: until the stock/price
tools land (#6) it says it can't confirm them. `LLM_MODEL` may be left empty to use the
provider's default model (see `.env.example`).

### Docker
```bash
make docker                                   # CPU-only image, ~0.6 GB compressed
docker run --rm -p 8000:8000 shop-ops-agent
curl localhost:8000/health                    # {"status":"ok","version":"0.1.0"}
```

## Evals
```bash
RUN_LIVE_EVALS=1 make evals
```
Golden set lives in `tests/evals/golden.jsonl`. Metrics: answer correctness,
faithfulness (LLM-as-judge), tool-selection accuracy, citation precision.

## Roadmap
See `docs/ISSUES.md`.
