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
IFS=',' read -r -a configured_roles <<< "${HEAR_POD_STACK_ROLES:-pipeline}"
for role in "${configured_roles[@]}"; do
  [[ -n "$role" ]] || continue
  replicas=$("$python" -c 'import json,os,sys; n=json.loads(os.environ.get("HEAR_WORKER_REPLICAS","{}" )).get(sys.argv[1],1); assert type(n) is int and 1<=n<=10; print(n)' "$role")
  for replica in $(seq 1 "$replicas"); do
  role_python="/opt/hear-ai-v11/venvs/$role/bin/python"
  HEAR_WORKER_ROLE="$role" HEAR_WORKER_ID="simulation-$role-$replica" "$role_python" -m hear.entrypoints.consumer >"$root/logs/$role-$replica.log" 2>&1 &
  child=$!; children+=("$child")
  echo "Starting real-model worker: $role (PID $child)"
  ready=false
  for n in {1..180}; do
    if ! kill -0 "$child" 2>/dev/null; then tail -30 "$root/logs/$role-$replica.log"; exit 1; fi
    if "$python" -c 'import httpx,sys; d=httpx.get("http://127.0.0.1:8000/readyz",timeout=3).json(); sys.exit(0 if d.get("lanes",{}).get(sys.argv[1],{}).get("status")=="ready" and d.get("lanes",{}).get(sys.argv[1],{}).get("consumers",0)>=int(sys.argv[2]) else 1)' "$role" "$replica" 2>/dev/null; then ready=true; break; fi
    sleep 2
  done
  if [[ "$ready" != true ]]; then echo "Worker startup timed out: $role"; tail -30 "$root/logs/$role-$replica.log"; exit 1; fi
  echo "Ready: $role replica=$replica"
  done
done
echo "SIMULATION API READY: all four real-model job types on port 8000"
set +e
wait -n "${children[@]}"
status=$?
set -e
exit "${status:-1}"
