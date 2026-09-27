#!/usr/bin/env bash
set -euo pipefail

if ! command -v rabbitmq-server >/dev/null || ! command -v rabbitmq-diagnostics >/dev/null; then
  printf '%s\n' 'Install rabbitmq-server before starting the Pod runtime.' >&2
  exit 1
fi

if runuser -u rabbitmq -- rabbitmq-diagnostics -q ping >/dev/null 2>&1 && timeout 1 bash -c 'exec 3<>/dev/tcp/127.0.0.1/5672' >/dev/null 2>&1; then
  exit 0
fi

runuser -u rabbitmq -- rabbitmq-server -detached
for _ in $(seq 1 60); do
  if runuser -u rabbitmq -- rabbitmq-diagnostics -q ping >/dev/null 2>&1 && timeout 1 bash -c 'exec 3<>/dev/tcp/127.0.0.1/5672' >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done

printf '%s\n' 'RabbitMQ did not become ready on 127.0.0.1:5672.' >&2
exit 1
