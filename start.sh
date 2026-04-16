#!/bin/bash
set -e

chattr -i /etc/resolv.conf 2>/dev/null || true
printf "nameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 1.1.1.1\n" > /etc/resolv.conf
chattr +i /etc/resolv.conf

echo "[START] DNS locked to 8.8.8.8, 8.8.4.4, 1.1.1.1"
echo "[START] Testing resolution..."
nslookup media.hear.surf > /dev/null 2>&1 && echo "[START] media.hear.surf OK" || echo "[START] WARNING: media.hear.surf still failing"
nslookup api.hear.surf   > /dev/null 2>&1 && echo "[START] api.hear.surf OK"   || echo "[START] WARNING: api.hear.surf still failing"

cd /workspace/hear-ai
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
