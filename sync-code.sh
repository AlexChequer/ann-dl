#!/usr/bin/env bash
# Regenerate the .py from the notebook, so the report's --8<-- include and the
# "runs from a clean checkout" criterion stay satisfied while you work in .ipynb.
set -euo pipefail
cd "$(dirname "$0")"
for nb in docs/exercises/*/code/*.ipynb; do
  [ -e "$nb" ] || continue
  .venv/bin/jupytext --to py:percent "$nb" -o "${nb%.ipynb}.py"
done
