#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$ROOT/.env"
  set +a
fi

eval "$(python3 <<'PY'
import os
import shlex
from urllib.parse import urlparse, unquote

raw = (os.environ.get("DATABASE_URL") or "").strip()
if not raw:
    raise SystemExit("DATABASE_URL is empty — set it in .env")

norm = raw.replace("postgresql+psycopg2://", "postgresql://", 1)
parsed = urlparse(norm)
user = unquote(parsed.username or "")
dbname = (parsed.path or "").lstrip("/").split("?")[0]
host = parsed.hostname or ""
port = str(parsed.port or 5432)

psql_url = raw.replace("postgresql+psycopg2://", "postgresql://", 1)
print(f"export DATABASE_URL={shlex.quote(raw)}")
print(f"export PSQL_URL={shlex.quote(psql_url)}")
print(f"export POSTGRES_USER={shlex.quote(user)}")
print(f"export POSTGRES_DB={shlex.quote(dbname)}")
print(f"export POSTGRES_HOST={shlex.quote(host)}")
print(f"export POSTGRES_PORT={shlex.quote(port)}")
PY
)"
