#!/bin/bash
set -e

WORKSPACE=/workspace/hear-ai
SUPERVISOR_CONF=/etc/supervisor/conf.d/hear-ai.conf
LOG_DIR=$WORKSPACE/logs
LOG_OUT=$LOG_DIR/hear-ai.out.log
LOG_ERR=$LOG_DIR/hear-ai.err.log
VENV=$WORKSPACE/venv

echo "[SETUP] Locking DNS..."
chattr -i /etc/resolv.conf 2>/dev/null || true
printf "nameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 1.1.1.1\n" > /etc/resolv.conf
chattr +i /etc/resolv.conf 2>/dev/null || true

nslookup media.hear.surf > /dev/null 2>&1 && echo "[DNS] media.hear.surf OK" || echo "[DNS] WARNING: media.hear.surf unreachable"
nslookup api.hear.surf   > /dev/null 2>&1 && echo "[DNS] api.hear.surf OK"   || echo "[DNS] WARNING: api.hear.surf unreachable"

echo "[SETUP] Installing system audio libraries and tools..."
apt-get update -qq && apt-get install -y -qq \
    supervisor \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-dev \
    libsox-fmt-all \
    dnsutils

mkdir -p $LOG_DIR

echo "[SETUP] Setting up Python virtual environment..."
cd $WORKSPACE

if [ ! -d "$VENV" ]; then
    python3 -m venv venv
    source $VENV/bin/activate
    echo "[SETUP] Installing Python dependencies..."
    pip install --no-cache-dir -q -r requirements.txt
else
    source $VENV/bin/activate
    echo "[SETUP] Virtual environment already exists, skipping install."
fi

echo "[SETUP] Writing Supervisor config..."
cat > $SUPERVISOR_CONF <<EOF
[program:hear-ai]
command=$VENV/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
directory=$WORKSPACE
autostart=true
autorestart=true
startretries=999
stderr_logfile=$LOG_ERR
stdout_logfile=$LOG_OUT
environment=PATH="$VENV/bin:%(ENV_PATH)s"
EOF

echo "[SETUP] All done. Starting Supervisor..."
echo "[LOGS] Live logs: tail -f $LOG_OUT"
exec supervisord -n
