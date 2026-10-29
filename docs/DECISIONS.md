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

## 2026-10-26 — Hybrid retrieval: RRF over Chroma cosine + BM25; English names in the data
Reciprocal-rank fusion (k=60) merges the two rankings without normalising their
incomparable scores. BM25 (`rank-bm25`, already a dependency; untyped, so one mypy
override) is built from the chunks stored in Chroma, so both rank the same corpus. English
queries against Bulgarian-only chunks missed (#4 comment): we fix that on the data side
with an optional `name_en` feed column appended to the chunk, not by translating queries
(no extra model call, no translation errors). Ranking quality is tested with the real
model in `tests/integration/` (CI caches the model); unit tests use fakes.

## 2026-10-27 — /chat: provider adapters, grounding and citations
- **Per-provider default model** when `LLM_MODEL` is empty (openai `gpt-4o-mini`,
  anthropic `claude-haiku-5-5`, ollama `llama3.2`), so switching `LLM_PROVIDER` alone
  never sends one vendor's model name to another; an explicit `LLM_MODEL` wins.
- **Claude adapter** follows the current API: no sampling params (rejected), content read
  by block type (Haiku 5.5 thinks by default), and `stop_reason == "refusal"` checked
  before content. Haiku has no server-side fallback.
- **No `effort` on Haiku:** a live call to `claude-haiku-5-5` returned 400 *"This model
  does not support the effort parameter"* (2026-10-27), although Anthropic's reference
  docs list effort for Haiku 5.5. The adapter never sends `output_config`; thinking depth
  is the model default. Revisit if we switch to an Opus/Sonnet model or the API changes.
- **Ollama via its OpenAI-compatible endpoint** with the `openai` SDK: no new dependency,
  one message/tool mapping for two providers.
- **Reply language is decided in code** and stated in the prompt, so it's deterministic
  and unit-testable: English only for Latin script containing unambiguous English words;
  Bulgarian otherwise, including Bulgarian typed in Latin letters ("imate li boiler").
- **Citations are verified:** only retrieved products whose code the answer cites. The
  parser accepts `[CODE: name]` and `[sku: CODE: name]` (the live model copied the word
  "sku" from an abstract instruction; the prompt now shows a fictitious concrete example
  instead, so the model can't mistake it for a real product to cite).
- **Prices/stock and invented products are enforced in code, not only by the prompt.** An
  answer with a money amount or a stock count (no tool can supply one before #6), or one
  citing a product the search didn't return, is replaced with a safe reply; the reason is
  on `ChatResult.guardrail` and logged. From #6 on, allow figures only when they appear
  verbatim in that turn's tool results. Availability claims without numbers ("наличен е")
  are not caught by this guard; the evals cover them.
- **Failures are 503s, never bare 500s:** config/index problems -> generic "not
  configured" (details only in server logs), provider outages -> "unavailable" with
  `Retry-After`. The service builds the LLM and opens the index before loading the
  embedding model, because `lru_cache` retries a failed build on every request.

## 2026-10-27 — Agent tools: registry, audiences, live catalog
- **`pytest-cov` (dev):** measures the "100% unit coverage of tool logic" criterion;
  `make test` (which CI now calls) fails below 100% line+branch on `agent/tools`.
- **Audiences:** `compare_supplier_price` exposes cost prices, so it is `operator`-only. The
  registry lists and runs tools per audience; a customer calling it gets the same
  "unknown tool" error as for a tool that doesn't exist.
- **The registry never raises:** unknown tool, invalid arguments (inputs forbid extra
  fields) or a crashing tool come back as a typed `ToolError` for the model to read.
- **Supplier gap = (ours − supplier) / ours × 100**, in BGN (the only currency both prices
  share), strictly greater than the threshold, computed in `Decimal`.
- **Price/stock come from the feed CSV (`CATALOG_PATH`)**, re-read when its mtime changes;
  a rejected new feed keeps the last good one. A shop DB/API can replace `Catalog` later.

## 2026-10-29 — Agent loop and session memory (#7)
- **Loop:** an iteration is one LLM call; at most 5. If the 5th call still asks for tools,
  they are not run (nothing would read the results) and the user gets a fixed "couldn't
  finish" reply. Tools run sequentially off the event loop; all results of one assistant
  turn go back together (one user message on Claude).
- **Verbatim results:** the tool message content is exactly `result.model_dump_json()`.
- **Thinking replay:** an assistant turn keeps the provider's raw content blocks and Claude
  gets them back unchanged within the turn, which is append-only.
- **Memory = text turns only** (question + the answer shown, guarded replies included,
  plus the products that answer verifiably cited, so follow-ups may cite them again),
  last 10 messages per session, LRU-capped at 10,000 sessions. **Session ids are issued
  by the server** (`secrets.token_urlsafe`); an unknown or client-chosen id (e.g. "demo")
  starts a new session, so nobody can read another conversation by guessing an id. Tool traffic and thinking
  blocks never enter memory: a sliding window changes the history every turn, which would
  invalidate replayed thinking blocks, and stale prices would sit in context. Upgrade path:
  Redis list per session (`RPUSH` + `LTRIM -10 -1` + `EXPIRE`), shared across workers.
- **Guardrails use this turn's tool results as evidence, by kind:** a EUR amount must equal a
  `price_eur`, a BGN amount a `price_bgn`, a stock count a `stock_qty` (`189` = `189.00`;
  a rounded `370 лв`, `4 EUR` where 4 is the stock, or the EUR price labelled `лв` are all
  blocked; `$` and cent amounts are never backed). A cited product must come from
  retrieval, a tool, or an earlier verified answer in the same session.
- **/chat is `customer`-only** until operators are authenticated; the supplier tool is
  neither offered nor executable there.
