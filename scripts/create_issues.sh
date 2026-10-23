#!/usr/bin/env bash
# Creates milestones + issues from docs/ISSUES.md in the current GitHub repo.
# Requires: gh (authenticated), run from the repo root after `git push` of main.
set -euo pipefail

for m in "M1 Foundation" "M2 RAG" "M3 Agent" "M4 Evals" "M5 Ship"; do
  gh api -X POST "repos/{owner}/{repo}/milestones" -f title="$m" >/dev/null 2>&1 || true
done

for l in "foundation:0E8A16" "rag:1D76DB" "agent:5319E7" "evals:D93F0B" "ship:FBCA04"; do
  gh label create "${l%%:*}" --color "${l##*:}" --force >/dev/null
done

# Parse docs/ISSUES.md: each "## #N Title — Mx" starts an issue; body runs until next "## #" or "---".
awk '
  /^## #[0-9]+ / {
    if (title != "") { print title "\t" ms "\t" body > "/tmp/_issue_" n ".txt" }
    n++; body = ""
    line = $0; sub(/^## #[0-9]+ /, "", line)
    split(line, parts, " — "); title = parts[1]; ms = parts[2]
    next
  }
  /^---/ { if (title != "") { print title "\t" ms "\t" body > "/tmp/_issue_" n ".txt"; title = "" } next }
  title != "" { body = body $0 "\n" }
' docs/ISSUES.md

for f in $(ls /tmp/_issue_*.txt | sort -V); do
  title=$(head -1 "$f" | cut -f1)
  ms=$(head -1 "$f" | cut -f2)
  body=$(tail -n +2 "$f")
  case "$ms" in
    M1*) label=foundation; ms="M1 Foundation";;
    M2*) label=rag;        ms="M2 RAG";;
    M3*) label=agent;      ms="M3 Agent";;
    M4*) label=evals;      ms="M4 Evals";;
    *)   label=ship;       ms="M5 Ship";;
  esac
  echo "Creating: $title [$ms]"
  gh issue create --title "$title" --body "$body" --milestone "$ms" --label "$label" >/dev/null
  rm -f "$f"
done
echo "Done. Check: gh issue list"
