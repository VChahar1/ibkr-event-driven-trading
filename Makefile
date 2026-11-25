PY := .venv/bin/python
.DEFAULT_GOAL := help

.PHONY: help venv up down topics test test-integration test-ib lint \
        run-adapter run-strategy run-reconciler run-recorder status \
        baseline kill-on kill-off replay

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | column -t -s "$$(printf '\t')"

venv: ## Create .venv with the pinned dependencies
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-eventbot.txt

up: ## Start Kafka and Redis, wait until healthy, create topics
	docker compose up -d --wait
	$(PY) scripts/create_topics.py

down: ## Stop Kafka and Redis (data volumes are kept)
	docker compose down

topics: ## Create topics (safe to repeat)
	$(PY) scripts/create_topics.py

test: ## Unit tests: no Docker and no IB needed
	$(PY) -m pytest

test-integration: ## Tests against real Kafka and Redis (needs `make up`)
	$(PY) -m pytest -m integration

test-ib: ## Read-only tests against the paper Gateway. Places no orders.
	$(PY) -m pytest -m ib_paper

lint: ## ruff and mypy
	.venv/bin/ruff check .
	.venv/bin/mypy .

run-adapter: ## Gateway adapter: the only process that talks to IB
	$(PY) -m eventbot.adapter.main

run-strategy: ## Strategy service (strategy.mode in config/eventbot.yaml, dry by default)
	$(PY) -m eventbot.strategy_service

run-reconciler: ## Reconciler: halts on any divergence
	$(PY) -m eventbot.reconciler

run-recorder: ## Recorder: all topics to Parquet
	$(PY) -m eventbot.recorder

status: ## One-screen health report
	$(PY) -m eventbot.status

baseline: ## Adopt the latest broker snapshot as the reconciliation baseline (asks to confirm)
	$(PY) scripts/adopt_baseline.py

kill-on: ## Set the kill switch
	$(PY) scripts/kill_switch.py on --reason "manual via make"

kill-off: ## Clear the kill switch (only after you have looked at why it was set)
	$(PY) scripts/kill_switch.py off

replay: ## Replay a recorded session through the system (FakeBroker only, never IB)
	$(PY) -m eventbot.replay
