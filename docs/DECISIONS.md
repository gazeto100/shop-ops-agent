# Decisions (append-only)

## 2026-10-23 — No LangChain / LlamaIndex
Build retrieval and the tool loop directly. Goal is to demonstrate understanding of
the primitives in interviews; frameworks hide exactly the parts that get asked about.

## 2026-10-23 — Local embeddings (multilingual-e5-small)
Bulgarian product names; zero indexing cost; runs on CPU in CI.

## 2026-10-23 — ChromaDB over pgvector
Single-process, no Postgres to run in CI. Documented upgrade path: pgvector when the
store needs concurrent writers.
