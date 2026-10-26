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

## 2026-10-24 — `numpy` as a direct dependency
`embeddings.py` returns typed `NDArray[float32]` vectors (what ChromaDB accepts). numpy
was already installed via sentence-transformers; declaring it stops us depending on a
transitive package.

## 2026-10-24 — Ingest syncs the index to the feed; price/stock never indexed
Ingest validates the whole CSV first and writes nothing if any row is invalid (all errors
reported). Then it upserts by sku and deletes skus missing from the feed, so retrieval
never suggests discontinued products. An empty feed is rejected, since syncing it would
wipe the index. Price and stock are kept out of embedded text *and* metadata: they change
daily and tools read them live (#6).
