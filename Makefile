WORKSPACE=/workspace/hear-ai
LOG_OUT=$(WORKSPACE)/logs/hear-ai.out.log
LOG_ERR=$(WORKSPACE)/logs/hear-ai.err.log

N ?= 200

.PHONY: boot start bootstrap-logs restart stop logs errors status install clean \
	env-check migrate clean-temp clean-temp-purge psql shell \
	errors-tail errors-head errors-cat errors-clear

boot:
	bash $(WORKSPACE)/start.sh

start:
	mkdir -p $(WORKSPACE)/logs
	bash $(WORKSPACE)/start.sh > $(WORKSPACE)/logs/bootstrap.log 2>&1
	@echo "Boot complete — run 'make status' or 'make logs'"

bootstrap-logs:
	tail -f $(WORKSPACE)/logs/bootstrap.log

restart:
	supervisorctl restart hear-ai 2>/dev/null || (make start && echo "Supervisor was not running — started fresh")

stop:
	supervisorctl stop hear-ai

status:
	supervisorctl status hear-ai

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
	@test -n "$$DATABASE_URL" || (echo "DATABASE_URL is empty" && exit 1)
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
