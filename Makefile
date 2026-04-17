WORKSPACE=/workspace/hear-ai
LOG_OUT=$(WORKSPACE)/logs/hear-ai.out.log
LOG_ERR=$(WORKSPACE)/logs/hear-ai.err.log

.PHONY: start restart stop logs status install clean

start:
	mkdir -p $(WORKSPACE)/logs
	nohup bash $(WORKSPACE)/start.sh > $(WORKSPACE)/logs/bootstrap.log 2>&1 &
	@echo "Booting in background — run 'make bootstrap-logs' to watch install progress"

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

install:
	source $(WORKSPACE)/venv/bin/activate && pip install -r requirements.txt

clean:
	rm -rf $(WORKSPACE)/venv
	rm -f $(LOG_OUT) $(LOG_ERR)
