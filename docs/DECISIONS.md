# Decisions (append-only)

## 2026-10-23 — No LangChain / LlamaIndex
Build retrieval and the tool loop directly. Goal is to demonstrate understanding of
the primitives in interviews; frameworks hide exactly the parts that get asked about.

## 2026-10-23 — Local embeddings (multilingual-e5-small)
Bulgarian product names; zero indexing cost; runs on CPU in CI.

## 2026-10-23 — ChromaDB over pgvector
Single-process, no Postgres to run in CI. Documented upgrade path: pgvector when the
store needs concurrent writers.

## 2026-10-23 — CPU-only torch on Linux (direct `torch` dependency)
PyPI's Linux torch wheel pulls ~3 GB of CUDA libraries; e5-small embeddings run fine on
CPU and the image must stay < 1.5 GB (#2). `torch` is listed directly only so
`[tool.uv.sources]` can route it to the PyTorch CPU index on Linux. Revisit if we ever
deploy to a GPU host.
