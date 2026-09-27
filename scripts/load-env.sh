#!/usr/bin/env bash
set -euo pipefail

readonly dotenv_path="${1:?usage: load-env.sh ENV_FILE}"

if [[ ! -f "$dotenv_path" ]]; then
  exit 0
fi

# Parse dotenv syntax instead of sourcing the file as shell code. This keeps
# JSON values such as BACKEND_REGISTRY_JSON intact and avoids shell expansion
# of credentials containing punctuation.
parser_python="${HEAR_ENV_PARSER_PYTHON:-/opt/hear-ai-v11/venvs/pipeline/bin/python}"
if [[ ! -x "$parser_python" ]]; then
  parser_python=python
fi
eval "$(ENV_FILE="$dotenv_path" "$parser_python" - <<'PY'
import os
import re
import shlex

from dotenv import dotenv_values

for key, value in dotenv_values(os.environ["ENV_FILE"]).items():
    if value is None:
        continue
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise SystemExit(f"invalid environment variable name: {key!r}")
    print(f"export {key}={shlex.quote(value)}")
PY
 )"
