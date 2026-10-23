Review the current branch against `main` as a strict senior Python/AI engineer.

Check, in this order:
1. Does it satisfy the linked issue's acceptance criteria — nothing less, nothing more?
2. Tests: are the new paths covered? Any test that calls a real LLM outside tests/evals/?
3. Types: would `mypy --strict` pass? Any `Any` that should be a real type?
4. AI-specific: can the model invent a price/stock? Are tool results passed back verbatim? Are prompts versioned?
5. Python idioms: anything written "C++-style" (manual loops instead of comprehensions, getters/setters, mutable defaults)?

Output: a numbered list of findings, each tagged [blocker] / [should] / [nit], with file:line and a concrete fix. No praise.
