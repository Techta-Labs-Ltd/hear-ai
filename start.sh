#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

WORKSPACE="${WORKSPACE:-/workspace/hear-ai}"
SUPERVISOR_CONF=/etc/supervisor/conf.d/hear-ai.conf
LOG_DIR=$WORKSPACE/logs
LOG_OUT=$LOG_DIR/hear-ai.out.log
LOG_ERR=$LOG_DIR/hear-ai.err.log
VENV=$WORKSPACE/venv

cd "$WORKSPACE"
if [ -f "$WORKSPACE/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  # .env is shell-sourced: wrap values that contain spaces in double quotes (e.g. HIGGS_AUDIO_SYSTEM_PROMPT).
  . "$WORKSPACE/.env"
  set +a
fi

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║          HEAR AI  —  Boot Sequence       ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo -e "  ${CYAN}WORKSPACE=${WORKSPACE}${RESET}"
echo ""

echo -e "${YELLOW}[1/8] Configuring DNS...${RESET}"
chattr -i /etc/resolv.conf 2>/dev/null || true
printf "nameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 1.1.1.1\n" > /etc/resolv.conf
chattr +i /etc/resolv.conf 2>/dev/null || true
nslookup media.hear.surf > /dev/null 2>&1 && echo -e "  ${GREEN}✓ media.hear.surf reachable${RESET}" || echo -e "  ${RED}✗ media.hear.surf unreachable (non-fatal)${RESET}"
nslookup api.hear.surf   > /dev/null 2>&1 && echo -e "  ${GREEN}✓ api.hear.surf reachable${RESET}"   || echo -e "  ${RED}✗ api.hear.surf unreachable (non-fatal)${RESET}"

echo ""
echo -e "${YELLOW}[2/8] Installing system audio libraries...${RESET}"
apt-get update -qq
apt-get install -y -qq supervisor ffmpeg libsndfile1 sox libsox-dev libsox-fmt-all dnsutils curl
echo -e "  ${GREEN}✓ supervisor, ffmpeg, libsndfile1, sox, libsox-dev, libsox-fmt-all, dnsutils, curl${RESET}"

mkdir -p $LOG_DIR

echo ""
echo -e "${YELLOW}[3/8] Installing Python dependencies (system-wide)...${RESET}"
pip install --no-cache-dir -r requirements.txt
echo -e "  ${GREEN}✓ All packages installed${RESET}"

echo ""
echo -e "${YELLOW}[4/8] Validating environment and PostgreSQL...${RESET}"
if [ -z "${DATABASE_URL:-}" ]; then
  echo -e "  ${RED}✗ DATABASE_URL is not set (required for PostgreSQL)${RESET}"
  exit 1
fi
if [ "${AI_SERVICE_SECRET:-change-me}" = "change-me" ]; then
  echo -e "  ${RED}✗ AI_SERVICE_SECRET must not be the default change-me${RESET}"
  exit 1
fi
if [ -z "${B2_KEY_ID:-}" ]; then
  echo -e "  ${YELLOW}⚠ B2_KEY_ID is empty (uploads will fail until set)${RESET}"
fi
if [[ "$DATABASE_URL" =~ :PORT(/|\?|#|$) ]]; then
  echo -e "  ${YELLOW}⚠ DATABASE_URL uses the literal \`:PORT\` (often copied from old docs). Replacing with :5432 for this process tree.${RESET}"
  echo -e "  ${YELLOW}  Edit ${WORKSPACE}/.env to use a numeric port, e.g. postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME?sslmode=require${RESET}"
  DATABASE_URL="${DATABASE_URL//:PORT/:5432}"
fi
if [[ "$DATABASE_URL" == *"@db.example.com:"* ]] || [[ "$DATABASE_URL" == *"@HOST:"* ]]; then
  echo -e "  ${RED}✗ DATABASE_URL still uses a documentation placeholder host (db.example.com or HOST).${RESET}"
  echo -e "  ${YELLOW}  Set the real PostgreSQL hostname or IP in ${WORKSPACE}/.env (see .env.example comments).${RESET}"
  exit 1
fi
export DATABASE_URL
if ! python3 -c "
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from app.config import settings
try:
    e = create_engine(settings.DATABASE_URL)
    with e.connect() as c:
        c.execute(text('select 1'))
    e.dispose()
except OperationalError as err:
    print('  PostgreSQL connection failed:', err.orig or err, file=sys.stderr)
    sys.exit(1)
"; then
  echo -e "  ${RED}✗ PostgreSQL unreachable (see message above). Fix DATABASE_URL in ${WORKSPACE}/.env${RESET}"
  exit 1
fi
echo -e "  ${GREEN}✓ PostgreSQL reachable${RESET}"

echo ""
echo -e "${YELLOW}[5/8] Database schema, temp dirs, and sweep...${RESET}"
python3 -c "from app.models.database import init_db; init_db(); print('  schema OK')"
mkdir -p "${HEAR_TMP_DIR:-/tmp/hear-ai}/jobs"
python3 -m app.tools.clean_temp --mode startup || true
echo -e "  ${GREEN}✓ Temp directories ready${RESET}"

echo ""
echo -e "${YELLOW}[6/8] Verifying FastAPI is installed...${RESET}"
python3 -c "import fastapi; print('  FastAPI', fastapi.__version__)"
echo -e "  ${GREEN}✓ FastAPI OK${RESET}"

echo ""
echo -e "${YELLOW}[7/8] Writing Supervisor config...${RESET}"
cat > $SUPERVISOR_CONF <<EOF
[program:hear-ai]
command=python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
directory=$WORKSPACE
autostart=true
autorestart=true
startretries=999
startsecs=20
stopasgroup=true
killasgroup=true
stopsignal=TERM
environment=PYTHONUNBUFFERED=1,GIT_PYTHON_REFRESH=quiet
stderr_logfile=$LOG_ERR
stdout_logfile=$LOG_OUT
EOF
echo -e "  ${GREEN}✓ Supervisor config written${RESET}"

echo ""
echo -e "${YELLOW}[8/8] Launching Hear AI server...${RESET}"
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  ✅  Server Starting on port 8000        ║${RESET}"
echo -e "${CYAN}${BOLD}║  📋 Logs:    make logs                   ║${RESET}"
echo -e "${CYAN}${BOLD}║  🔄 Restart: make restart                ║${RESET}"
echo -e "${CYAN}${BOLD}║  🛑 Stop:    make finish                 ║${RESET}"
echo -e "${CYAN}${BOLD}║  ▶ Full boot: make / make up             ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""

if pgrep -x supervisord >/dev/null 2>&1; then
  supervisorctl reread >/dev/null 2>&1 || true
  supervisorctl update >/dev/null 2>&1 || true
  supervisorctl restart hear-ai >/dev/null 2>&1 || supervisorctl start hear-ai >/dev/null 2>&1
  echo -e "${GREEN}✓ Reused existing supervisord and (re)started hear-ai${RESET}"
  exit 0
fi

supervisord
sleep 1
supervisorctl reread >/dev/null 2>&1 || true
supervisorctl update >/dev/null 2>&1 || true
supervisorctl start hear-ai >/dev/null 2>&1 || true
echo -e "${GREEN}✓ supervisord started in daemon mode${RESET}"
