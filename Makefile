# Common commands. Training and experiments: see README.md.
PY ?= .venv/bin/python

.PHONY: help release serve test lint docker-build docker-run up monitoring mlflow-import mlflow-ui

help:            ## list commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

release:         ## package the best checkpoints into checkpoints/ (with release gate) + demo examples
	$(PY) scripts/package_models.py

serve:           ## run the demo + API locally on http://127.0.0.1:7860
	$(PY) -m app.server

test:            ## unit + API tests (no dataset needed)
	$(PY) -m pytest

lint:            ## ruff
	$(PY) -m ruff check .

docker-build:    ## build the serving image (needs checkpoints/)
	docker build -t polyp-seg:latest .

docker-run:      ## run the serving image on port 7860
	docker run --rm -p 7860:7860 polyp-seg:latest

up:              ## docker compose: demo + API
	docker compose up -d --build

monitoring:      ## docker compose: demo + Prometheus (:9090) + Grafana (:3000)
	docker compose --profile monitoring up -d --build

mlflow-import:   ## import finished runs and the current release into MLflow (mlruns/)
	$(PY) scripts/log_to_mlflow.py --release checkpoints

mlflow-ui:       ## browse MLflow on http://127.0.0.1:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
