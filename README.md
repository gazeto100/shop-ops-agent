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
  -d '{"message":"Има ли наличен бойлер 80 литра и каква е цената?"}'
```

`/chat` returns `{"session_id": "...", "answer": "...", "citations": [{"sku": "...", "name": "..."}], "tool_calls": [{"name": "...", "arguments": {...}}]}`.
The answer is grounded in the retrieved catalog excerpt and the agent's tools
(`search_products`, `check_stock_and_price`; the supplier-price tool is operator-only and never
offered on `/chat`), and written in the user's language (Bulgarian unless the message is in
English). The agent runs up to 5 model calls per message, then stops with a clear message.
Prices and stock come only from `check_stock_and_price` in the same turn: an answer stating a
figure no tool returned, or citing a product neither search nor a tool returned, is replaced
by a safe reply. The response's `session_id` is issued by the server: send it back with the
next message for follow-ups (the last 10 messages are remembered, per process, lost on
restart). An id the server didn't issue just starts a new session. A misconfigured service or an
unavailable LLM provider returns `503` (with `Retry-After`). `LLM_MODEL` may be left empty to
use the provider's default model (see `.env.example`).

> **Windows (Git Bash):** an inline `curl -d '…'` with Cyrillic text is sent in the console
> code page, not UTF-8, so the server receives garbled text. Put the body in a UTF-8 file
> and send it as-is:
>
> ```bash
> printf '%s' '{"message":"Има ли наличен бойлер 80 литра?"}' > chat.json
> curl -X POST localhost:8000/chat -H 'content-type: application/json' --data-binary @chat.json
> ```

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
