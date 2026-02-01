---
name: commit-precommit-retry
description: Commits staged changes; if pre-commit fails (e.g. Black), runs Black and recommits. Use when the user asks to commit changes, commit with pre-commit, or fix pre-commit and recommit after formatting.
---

# Commit with Pre-commit Retry

## Workflow

1. **Commit** staged changes with a message (user-provided or inferred from diff).
   ```bash
   git commit -m "<message>"
   ```

2. **If the commit fails** and the output indicates a formatter (e.g. Black) failed:
   - Run Black on the repo (or on the files that were modified):
     - In a uv project: `uv run black .` or `uv run black <paths>`
     - Otherwise: `black .` or `black <paths>`
   - Re-stage and commit again:
     ```bash
     git add -u
     git commit -m "<same message>"
     ```

3. **If it fails again** for another hook (e.g. isort, flake8), run that tool, re-stage, and recommit. Repeat until the commit succeeds or the failure is unrelated to auto-fixable hooks.

## Notes

- Prefer the same commit message on retry so the change set stays one commit.
- If the user did not provide a message, suggest one from the staged diff (e.g. short summary of changed files or first line of diff).
- Do not use `--no-verify` to skip hooks unless the user explicitly asks to skip them.
