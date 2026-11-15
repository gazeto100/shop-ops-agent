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
  *Superseded 2026-11-15: the 400 came from `claude-haiku-4-5`, not Haiku 5.5 (see below).*
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

## 2026-11-03 — Eval runner and golden set (#8)
- **Tool selection is scored as recall of the tools an answer requires** (extras listed,
  not penalised). `[]` = no tool may be called; `null` = not scored, because the retrieved
  catalog excerpt often makes `search_products` unnecessary and requiring it would punish
  reasonable behaviour. Existing cases were adjusted to "required tools only".
- **Facts and forbidden strings are case-insensitive string checks** (deterministic, cheap)
  where numbers match only as whole numbers: plain substrings flagged honest answers
  ("0 EUR" inside "24.90 EUR") and let "36" pass on "360". The answer language (`lang`)
  is scored too, by word counts that ignore citations, units and brands. Semantic
  faithfulness is the LLM-as-judge in #9.
- **The eval runs the real `ChatService`, guardrails included**, so it scores what a user
  would see, against a fresh index of the sample catalog the cases are written for.
- **Results (`evals/results-<date>.json`) are gitignored run artifacts**; #9 can publish
  them from CI. The eval fails only on errored cases; score thresholds are #9's gate.

## 2026-11-15 — Effort by model; correction to "No `effort` on Haiku" (#5)
- **The #5 note blamed the wrong model.** The live run that returned 400 *"This model does
  not support the effort parameter"* used `LLM_MODEL=claude-haiku-4-5` from `.env` (the #8
  eval report shows the same model), not the code default `claude-haiku-5-5`.
- **Which models take `output_config.effort`** (Anthropic models overview and migration
  guide, checked 2026-11-15; the installed SDK 2.1.0 types the levels
  `low`…`max` but doesn't say which models accept them): Haiku 5.5 (default `medium`),
  Sonnet 4.6 / 5 / 5.5, Opus 4.5 (`low`–`high` only) / 4.6 / 4.7 / 4.8 / 5 / 5.5,
  Fable / Mythos 5.x. **Haiku 4.5 and Sonnet 4.5 reject it.** Not re-verified live: no
  paid call was made for this change.
- **`AnthropicLLM` sends `effort: "low"` only to models on that allow-list** (family id or
  a `<family>-…` snapshot). Short catalog answers don't need deep thinking, and low
  means fewer tokens and less latency than Haiku 5.5's `medium` default. Unknown or
  older models get no `output_config`, so a new model id never 400s on effort; add it to
  `_EFFORT_MODELS` once its docs list effort. A `.env` still on `claude-haiku-4-5` keeps
  working exactly as before.

## 2026-11-15 — Prices per product; supplier questions via the tool (prompt 2026-11-15.1)
- **No price arithmetic, enforced in the prompt, not by loosening the guard.** Live evals
  g022/g023/g025 failed because the model wrote differences or totals; those figures are
  not in any tool result, so the price guard replaced the answer. Allowing computed
  figures would mean the guard verifying arithmetic, and a wrong sum would be exactly the
  invented price the guard exists to stop. The prompt now says: state each price as
  returned, never compute a difference, total or percentage, list the prices instead and
  say which product is cheaper.
- **Per-audience system prompt.** Operators get an extra section: supplier, cost or
  margin questions always call `compare_supplier_price` (g004 stopped calling it on
  `claude-haiku-5-5`), with `threshold_pct` 0 when none is named, and its figures are
  stated as returned. Customers' prompt doesn't name the internal tool at all. Each
  audience's prompt is still a constant, so prefix caching is unaffected.

## 2026-11-15 — Faithfulness judge and the eval gate (#9)
- **Judge model: `claude-sonnet-5-5`, not the agent's Haiku.** A model grading its own
  output tends to share its blind spots (and to rate its own phrasing kindly); a
  different, stronger model is a more independent check. Sonnet rather than Opus: the
  task is narrow (are these claims in this context?) and runs once per case on every
  main build. Configurable via `JUDGE_PROVIDER` / `JUDGE_MODEL`.
- **The judge answers through a `submit_verdict` tool call** with a pydantic-validated
  `{score 0–1, reason}`, so it works through the same provider layer as the agent, on any
  provider. A missing or invalid verdict errors the case instead of guessing a score.
- **It grades what the user saw** (guarded replies included) against what the agent was
  given that turn: the retrieved excerpt and each tool result as the model received it.
- **Gate:** pass rate ≥ 0.90, mean faithfulness ≥ 0.80, and no `must_not_contain`
  violation; all three must hold, and an unmeasured value fails. The issue asked for
  faithfulness ≥ 0.80; the pass-rate bar was added so the deterministic checks (tools,
  facts, citations, language) gate too.
- **CI `evals` runs the agent on Anthropic** (Haiku 5.5) with the Sonnet judge, so one
  secret, `ANTHROPIC_API_KEY`, covers both; `OPENAI_API_KEY` is no longer used there. The
  score table goes to the job summary (the job runs on `main` only, so there is no PR to
  comment on).
