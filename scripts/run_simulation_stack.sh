#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=/root/hear-ai-v11/simulation-20260928
source scripts/load-env.sh "$root/runtime.env"
export HEAR_ENV_FILE="$root/runtime.env" PYTHONPATH="$PWD"
python=/opt/hear-ai-v11/venvs/test/bin/python
children=()
cleanup() { for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT INT TERM
"$python" -m scripts.simulation_backend >"$root/logs/backend.log" 2>&1 &
children+=("$!")
for n in {1..30}; do
  if curl -fsS --cacert "$root/tls/cert.pem" https://127.0.0.1:18081/source/fish-reference.wav -o /dev/null; then break; fi
  sleep 1
done
export HTTP_HOST=0.0.0.0 HTTP_PORT=8000
"$python" -m hear.entrypoints.gateway >"$root/logs/gateway.log" 2>&1 &
children+=("$!")
for role in reconstruction pipeline transcription magic_clean_natural; do
  role_python="/opt/hear-ai-v11/venvs/$role/bin/python"
  HEAR_WORKER_ROLE="$role" HEAR_WORKER_ID="simulation-$role-01" "$role_python" -m hear.entrypoints.consumer >"$root/logs/$role.log" 2>&1 &
  child=$!; children+=("$child")
  echo "Starting real-model worker: $role (PID $child)"
  ready=false
  for n in {1..180}; do
    if ! kill -0 "$child" 2>/dev/null; then tail -30 "$root/logs/$role.log"; exit 1; fi
    if "$python" -c 'import httpx,sys; d=httpx.get("http://127.0.0.1:8000/readyz",timeout=3).json(); sys.exit(0 if d.get("lanes",{}).get(sys.argv[1],{}).get("status")=="ready" else 1)' "$role" 2>/dev/null; then ready=true; break; fi
    sleep 2
  done
  if [[ "$ready" != true ]]; then echo "Worker startup timed out: $role"; tail -30 "$root/logs/$role.log"; exit 1; fi
  echo "Ready: $role"
done
echo "SIMULATION API READY: all four real-model job types on port 8000"
set +e
wait -n "${children[@]}"
status=$?
set -e
exit "${status:-1}"
