# Evals

`golden.jsonl` — one case per line. Fields:
- `question` — user input
- `expected_tools` — tools the agent is expected to call (tool-selection accuracy)
- `expected_facts` — substrings that must appear in the final answer
- `must_not_contain` — substrings that must NOT appear (hallucination guard)
- `must_cite` — answer must include at least one catalog citation
- `lang` — expected answer language (default `bg`)
- `audience` — `customer` (default, the public /chat) or `operator` (staff; may use
  internal tools such as `compare_supplier_price`)

Issue #8 adds the runner; #9 adds the LLM-as-judge faithfulness score and the CI gate
(fail if mean score < 0.8 or any `must_not_contain` violated).

Target: 30–40 cases before calling the project "done". Add a case for every bug you fix.
