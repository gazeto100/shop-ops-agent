Run the eval suite and report.

1. `RUN_LIVE_EVALS=1 make evals`
2. Summarize: mean faithfulness, tool-selection accuracy, cases failed (id + reason).
3. For each failure, propose the smallest fix (prompt, retrieval, or tool) and whether it needs a new golden case.
Do not change any code in this command — report only.
