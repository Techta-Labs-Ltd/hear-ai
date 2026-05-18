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
import shlex
from app.config import settings

url = (settings.DATABASE_URL or "").strip()
if not url:
    raise SystemExit("DATABASE_URL is empty — set POSTGRES_* or DATABASE_URL in .env")

psql_url = url.replace("postgresql+psycopg2://", "postgresql://", 1)
print(f"export DATABASE_URL={shlex.quote(url)}")
print(f"export PSQL_URL={shlex.quote(psql_url)}")
print(f"export POSTGRES_USER={shlex.quote(settings.POSTGRES_USER)}")
print(f"export POSTGRES_DB={shlex.quote(settings.POSTGRES_DB)}")
print(f"export POSTGRES_HOST={shlex.quote(settings.POSTGRES_HOST)}")
print(f"export POSTGRES_PORT={shlex.quote(str(settings.POSTGRES_PORT))}")
PY
)"
