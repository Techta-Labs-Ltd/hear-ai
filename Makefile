# Hear AI — RunPod / bare-metal bootstrap
# Override install root:  make up WORKSPACE=/app
# Default `make` runs the full boot (same as start.sh end-to-end).

WORKSPACE ?= /workspace/hear-ai
export WORKSPACE

LOG_OUT := $(WORKSPACE)/logs/hear-ai.out.log
LOG_ERR := $(WORKSPACE)/logs/hear-ai.err.log
N ?= 200
HEALTH_URL ?= http://127.0.0.1:8000/health
WAIT_READY_SECS ?= 90

.DEFAULT_GOAL := up

.PHONY: help up up-bg all boot start finish down restart stop logs errors status \
	bootstrap-logs env-check migrate clean-temp clean-temp-purge psql shell \
	install clean errors-tail errors-head errors-cat errors-clear wait-ready

help: ## Show targets (default: `make` = full bootstrap via `up`)
	@echo "Hear AI — common targets"
	@echo ""
	@echo "  make / make up     Full bootstrap (DNS, apt, pip, .env, Postgres, temp sweep,"
	@echo "                     Supervisor, uvicorn). Same as start.sh, foreground output."
	@echo "  make up-bg         Same bootstrap, log to $(WORKSPACE)/logs/bootstrap.log"
	@echo "  make wait-ready    Poll $(HEALTH_URL) until OK (max $(WAIT_READY_SECS)s)"
	@echo "  make finish        Stop the app (supervisorctl stop)"
	@echo "  make restart       Restart hear-ai program (or full up if supervisor missing)"
	@echo "  make logs / errors Follow stdout / stderr"
	@echo "  make status        supervisorctl status"
	@echo "  make env-check     Quick DATABASE_URL + AI_SERVICE_SECRET check"
	@echo "  make migrate       init_db() only (schema + extensions)"
	@echo "  make clean-temp    Periodic temp sweep"
	@echo ""
	@echo "WORKSPACE=$(WORKSPACE) (override: make up WORKSPACE=/your/path)"

up: ## Full install + validate + start server (entire start.sh pipeline)
	bash $(WORKSPACE)/start.sh

all: up ## Alias for `up`

boot: up ## Alias for `up` (legacy name)

up-bg: ## Full bootstrap with output to logs/bootstrap.log (non-interactive)
	mkdir -p $(WORKSPACE)/logs
	bash $(WORKSPACE)/start.sh > $(WORKSPACE)/logs/bootstrap.log 2>&1
	@echo "Boot complete — $(WORKSPACE)/logs/bootstrap.log"
	@echo "Run: make status   or   make bootstrap-logs   or   make wait-ready"

start: up-bg ## Legacy alias (background bootstrap log)

bootstrap-logs:
	tail -f $(WORKSPACE)/logs/bootstrap.log

wait-ready: ## Wait until /health returns 200 (requires curl; max WAIT_READY_SECS)
	@secs=0; \
	while [ $$secs -lt $(WAIT_READY_SECS) ]; do \
	  if curl -sf "$(HEALTH_URL)" >/dev/null 2>&1; then \
	    echo "OK — $(HEALTH_URL)"; \
	    exit 0; \
	  fi; \
	  sleep 2; \
	  secs=$$((secs+2)); \
	done; \
	echo "Timeout waiting for $(HEALTH_URL)"; \
	exit 1

finish: down ## Stop app (alias)

down: ## Stop hear-ai under Supervisor
	supervisorctl stop hear-ai 2>/dev/null || true

restart:
	supervisorctl restart hear-ai 2>/dev/null || ( $(MAKE) up-bg && echo "Supervisor was not running — ran full bootstrap (up-bg)" )

stop: down ## Alias for `down`

status:
	supervisorctl status hear-ai 2>/dev/null || echo "supervisor not running or hear-ai not defined"

logs:
	tail -f $(LOG_OUT)

errors:
	tail -f $(LOG_ERR)

errors-tail:
	@test -f $(LOG_ERR) && tail -n $(N) $(LOG_ERR) || echo "No errors yet ($(LOG_ERR))"

errors-head:
	@test -f $(LOG_ERR) && head -n $(N) $(LOG_ERR) || echo "No errors yet ($(LOG_ERR))"

errors-cat:
	@test -f $(LOG_ERR) && cat $(LOG_ERR) || echo "No errors yet ($(LOG_ERR))"

errors-clear:
	@: > $(LOG_ERR) && echo "Cleared $(LOG_ERR)"

env-check:
	@test -n "$$DATABASE_URL" || (echo "DATABASE_URL is empty (export or put in $(WORKSPACE)/.env)" && exit 1)
	@test "$$AI_SERVICE_SECRET" != "change-me" || (echo "AI_SERVICE_SECRET must not be change-me" && exit 1)

migrate:
	cd $(WORKSPACE) && python3 -c "from app.models.database import init_db; init_db(); print('migrated')"

clean-temp:
	cd $(WORKSPACE) && python3 -m app.tools.clean_temp --mode periodic

clean-temp-purge:
	cd $(WORKSPACE) && python3 -m app.tools.clean_temp --mode purge --yes

psql:
	@psql "$$DATABASE_URL"

shell:
	cd $(WORKSPACE) && python3 -i -c "from app.models.database import SessionLocal, AiJob, AiTrackJob, AiTempFile; db = SessionLocal()"

install:
	source $(WORKSPACE)/venv/bin/activate && pip install -r requirements.txt

clean:
	rm -rf $(WORKSPACE)/venv
	rm -f $(LOG_OUT) $(LOG_ERR)
