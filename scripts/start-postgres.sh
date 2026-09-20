#!/usr/bin/env bash
set -euo pipefail

readonly project_root="/workspace/hear-ai"
readonly postgres_bin="/usr/lib/postgresql/16/bin"
readonly postgres_data="/postgres/data"
readonly postgres_socket="/postgres/socket"
readonly database_url="$(sed -n 's/^DATABASE_URL=//p' "${project_root}/.env")"

if [[ ! "$database_url" =~ ^postgresql\+psycopg2://hear:([^@]+)@127\.0\.0\.1:5432/hear$ ]]; then
  echo "DATABASE_URL must use the local hear PostgreSQL database" >&2
  exit 1
fi
readonly database_password="${BASH_REMATCH[1]}"

install -d -o postgres -g postgres -m 0700 "$postgres_data" "$postgres_socket"
if [[ ! -f "$postgres_data/PG_VERSION" ]]; then
  runuser -u postgres -- "$postgres_bin/initdb" \
    --pgdata="$postgres_data" \
    --auth-host=scram-sha-256 \
    --auth-local=trust \
    --no-instructions
  {
    echo "listen_addresses = '127.0.0.1'"
    echo "port = 5432"
    echo "unix_socket_directories = '$postgres_socket'"
  } >> "$postgres_data/postgresql.conf"
fi

runuser -u postgres -- "$postgres_bin/pg_ctl" --pgdata="$postgres_data" --wait start
role_exists="$(
  runuser -u postgres -- "$postgres_bin/psql" \
    --host="$postgres_socket" --port=5432 --dbname=postgres --tuples-only --no-align \
    --command="SELECT 1 FROM pg_roles WHERE rolname = 'hear';"
)"
if [[ "$role_exists" == "1" ]]; then
  runuser -u postgres -- "$postgres_bin/psql" \
    --host="$postgres_socket" --port=5432 --dbname=postgres --set=ON_ERROR_STOP=1 \
    --command="ALTER ROLE hear LOGIN PASSWORD '$database_password';"
else
  runuser -u postgres -- "$postgres_bin/psql" \
    --host="$postgres_socket" --port=5432 --dbname=postgres --set=ON_ERROR_STOP=1 \
    --command="CREATE ROLE hear LOGIN PASSWORD '$database_password';"
fi
database_exists="$(
  runuser -u postgres -- "$postgres_bin/psql" \
    --host="$postgres_socket" --port=5432 --dbname=postgres --tuples-only --no-align \
    --command="SELECT 1 FROM pg_database WHERE datname = 'hear';"
)"
if [[ "$database_exists" != "1" ]]; then
  runuser -u postgres -- "$postgres_bin/psql" \
    --host="$postgres_socket" --port=5432 --dbname=postgres --set=ON_ERROR_STOP=1 \
    --command="CREATE DATABASE hear OWNER hear;"
fi
runuser -u postgres -- "$postgres_bin/pg_ctl" --pgdata="$postgres_data" --wait stop

exec runuser -u postgres -- "$postgres_bin/postgres" -D "$postgres_data"
