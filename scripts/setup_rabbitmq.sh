#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  printf '%s\n' 'Run RabbitMQ setup as root.' >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y rabbitmq-server
install -m 0644 deploy/runtime/rabbitmq.conf /etc/rabbitmq/rabbitmq.conf
runuser -u rabbitmq -- rabbitmqctl shutdown >/dev/null 2>&1 || true
runuser -u rabbitmq -- rabbitmq-server -detached
ready=0
for _ in $(seq 1 60); do
  if runuser -u rabbitmq -- rabbitmq-diagnostics -q ping >/dev/null 2>&1 && timeout 1 bash -c 'exec 3<>/dev/tcp/127.0.0.1/5672' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  printf '%s\n' 'RabbitMQ did not become ready on 127.0.0.1:5672.' >&2
  exit 1
fi
