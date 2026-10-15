#!/usr/bin/env bash
# Repeatable automated checks. Always runs offline against mock services.
set -euo pipefail
cd "$(dirname "$0")"
if [ -d .venv ]; then . .venv/bin/activate; fi
export MOCK_SERVICES=1
python -m pytest -q "$@"
