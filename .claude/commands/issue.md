Work on GitHub issue #$ARGUMENTS.

1. Run `gh issue view $ARGUMENTS` and read it fully.
2. Restate the acceptance criteria as a checklist. If anything is ambiguous, ask me before coding.
3. Create a branch `issue-$ARGUMENTS-<short-slug>` from `main`.
4. Implement the smallest change that satisfies the criteria, following CLAUDE.md.
5. Add/adjust unit tests. If prompts, retrieval, or tools changed, add a golden-set case.
6. Run `make lint test` and fix everything.
7. Commit with a conventional message referencing the issue, e.g. `feat(rag): hybrid retrieval (#$ARGUMENTS)`.
8. Open a PR with `gh pr create --fill` using the PR template. In the "Python notes" section, explain any idiom a C++/PHP developer may not know.
