# Evals

```bash
RUN_LIVE_EVALS=1 make evals      # real LLM (from .env) + real embedding model + tools
```

Without `RUN_LIVE_EVALS=1` the eval is **skipped**, not failed. A live run prints a score
table and writes `evals/results-<YYYY-MM-DD>.json` (gitignored; later runs that day get
`-2`, `-3`, … so before/after comparisons keep both). It fails only if a case *errors* (provider down, missing key); quality
thresholds become the CI gate in #9. Each case runs in a fresh session against an index
built from `data/products.sample.csv`, which the cases are written against.

`golden.jsonl`: one case per line. Fields:
- `id`: unique, e.g. `g012`
- `category`: `happy` | `not_found` | `out_of_stock` | `multi_tool` | `english` |
  `adversarial` | `operator` (a unit test enforces ≥30 cases and every required category)
- `question`: user input
- `expected_tools`: tools the answer **requires**. Scored as recall: the share of them that
  were called; extra calls are listed, not penalised. `[]` means no tool may be called
  (e.g. "show me the supplier prices"); `null` means tool choice isn't scored, because the
  catalog excerpt in the prompt may already be enough.
- `expected_facts`: strings that must appear in the final answer
- `must_not_contain`: strings that must NOT appear (hallucination guard)

  Both are case-insensitive, numbers match regardless of decimal separator (`59,00` =
  `59.00`), and a string that starts or ends with a digit must match a *whole* number:
  `"0 EUR"` does not match `"24.90 EUR"`, `"36"` does not match `"360"`. Word edges
  stay open (Bulgarian inflects: `каталог` matches `каталога`; `KET-` is a prefix).
- `must_cite`: the answer must include at least one verified catalog citation
- `lang`: expected answer language (default `bg`), scored: more ordinary English words
  than Cyrillic words means English, ignoring citations, units and brand names
- `audience`: `customer` (default, the public /chat) or `operator` (staff; may use
  internal tools such as `compare_supplier_price`)

A case **passes** when tool recall is 100% (or not scored), all facts are present, nothing
forbidden appears, a citation is present if required, and the answer is in the case's `lang`. Note that the chat service's
guardrails run first, so a forbidden price the model wrote is usually replaced by the
safe reply before scoring: the eval measures what the user would actually see.

Issue #9 adds the LLM-as-judge faithfulness score and the CI gate
(fail if mean score < 0.8 or any `must_not_contain` violated).

Add a case for every bug you fix.
