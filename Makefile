WORKSPACE=/workspace/hear-ai
LOG_OUT=$(WORKSPACE)/logs/hear-ai.out.log
LOG_ERR=$(WORKSPACE)/logs/hear-ai.err.log

.PHONY: start restart stop logs status install clean

start:
	bash $(WORKSPACE)/start.sh

restart:
	supervisorctl restart hear-ai

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
