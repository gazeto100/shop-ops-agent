# shop-ops-agent

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

## Evals
```bash
RUN_LIVE_EVALS=1 make evals
```
Golden set lives in `tests/evals/golden.jsonl`. Metrics: answer correctness,
faithfulness (LLM-as-judge), tool-selection accuracy, citation precision.

## Roadmap
See `docs/ISSUES.md`.
