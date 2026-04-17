#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

WORKSPACE=/workspace/hear-ai
SUPERVISOR_CONF=/etc/supervisor/conf.d/hear-ai.conf
LOG_DIR=$WORKSPACE/logs
LOG_OUT=$LOG_DIR/hear-ai.out.log
LOG_ERR=$LOG_DIR/hear-ai.err.log
VENV=$WORKSPACE/venv

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║          HEAR AI  —  Boot Sequence       ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""

echo -e "${YELLOW}[1/6] Configuring DNS...${RESET}"
chattr -i /etc/resolv.conf 2>/dev/null || true
printf "nameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 1.1.1.1\n" > /etc/resolv.conf
chattr +i /etc/resolv.conf 2>/dev/null || true
nslookup media.hear.surf > /dev/null 2>&1 && echo -e "  ${GREEN}✓ media.hear.surf reachable${RESET}" || echo -e "  ${RED}✗ media.hear.surf unreachable (non-fatal)${RESET}"
nslookup api.hear.surf   > /dev/null 2>&1 && echo -e "  ${GREEN}✓ api.hear.surf reachable${RESET}"   || echo -e "  ${RED}✗ api.hear.surf unreachable (non-fatal)${RESET}"

echo ""
echo -e "${YELLOW}[2/6] Installing system audio libraries...${RESET}"
apt-get update -qq
apt-get install -y -qq supervisor ffmpeg libsndfile1 sox libsox-dev libsox-fmt-all dnsutils
echo -e "  ${GREEN}✓ supervisor, ffmpeg, libsndfile1, sox, libsox-dev, libsox-fmt-all, dnsutils${RESET}"

mkdir -p $LOG_DIR

echo ""
echo -e "${YELLOW}[3/6] Setting up Python virtual environment...${RESET}"
cd $WORKSPACE

if [ ! -d "$VENV" ]; then
    python3 -m venv venv
    source $VENV/bin/activate
    echo -e "  ${GREEN}✓ Virtual environment created${RESET}"
    echo ""
    echo -e "${YELLOW}[4/6] Installing Python dependencies...${RESET}"
    pip install --no-cache-dir -q -r requirements.txt
    echo -e "  ${GREEN}✓ All packages installed from requirements.txt${RESET}"
else
    source $VENV/bin/activate
    echo -e "  ${GREEN}✓ Virtual environment already exists — skipping install${RESET}"
    echo -e "${YELLOW}[4/6] Skipping pip install (cached)${RESET}"
fi

echo ""
echo -e "${YELLOW}[5/6] Writing Supervisor config...${RESET}"
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
echo -e "  ${GREEN}✓ Supervisor config written${RESET}"

echo ""
echo -e "${YELLOW}[6/6] Launching Hear AI server...${RESET}"
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  ✅  Server Starting on port 8000        ║${RESET}"
echo -e "${CYAN}${BOLD}║  📋 Logs: make logs                      ║${RESET}"
echo -e "${CYAN}${BOLD}║  🔄 Restart: make restart                ║${RESET}"
echo -e "${CYAN}${BOLD}║  🛑 Stop:    make stop                   ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""

exec supervisord -n
