# ================================================================
# XYZ MDM 3.0 — Model Training Pipeline Makefile
# Uses pip + setuptools (not Poetry)
# ================================================================
.PHONY: install run-local run-dev migrate train evaluate promote test lint clean help

install: ## Install all dependencies (warning: downloads ~3GB ML models on first run)
	pip install -e ".[dev]"

env-setup: ## Create .env.local from .env.local.example
	@if not exist .env.local (copy .env.local.example .env.local && echo .env.local created.) else (echo .env.local already exists.)

## Database
migrate: ## Run Alembic migrations (set POSTGRES_PASSWORD env var first)
	alembic upgrade head

migrate-create: ## Create new migration
	alembic revision --autogenerate -m "$(name)"

## Run
run-local: ## Run API server locally
	uvicorn src.main:app --host 0.0.0.0 --port 8110 --reload

run-dev: ## Run in dev mode
	uvicorn src.main:app --host 0.0.0.0 --port 8110 --reload --log-level debug

run: ## Run without reload
	uvicorn src.main:app --host 0.0.0.0 --port 8110

## ML Pipeline CLI
train: ## Trigger training run (usage: make train args="--entity-type customer")
	python -m src.cli train $(args)

evaluate: ## Evaluate a model run (usage: make evaluate args="--run-id <id>")
	python -m src.cli evaluate $(args)

promote: ## Promote a model to champion (usage: make promote args="--run-id <id>")
	python -m src.cli promote $(args)

## Test
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Code Quality
lint: ## Run linter
	ruff check src/ 2>nul || echo ruff not installed, skipping

fmt: ## Format code
	ruff check src/ --fix 2>nul || echo ruff not installed, skipping

## Docker
docker-build: ## Build Docker image
	docker build -t xyz-mdm/model-training-pipeline:3.0.0 .

compose-up: ## Start with docker-compose
	docker-compose up -d

compose-down: ## Stop docker-compose
	docker-compose down

## Clean
clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>nul || true
	find . -type f -name "*.pyc" -delete 2>nul || true
	rmdir /s /q .pytest_cache 2>nul || true
	rmdir /s /q htmlcov 2>nul || true
	rmdir /s /q src\model_training_pipeline.egg-info 2>nul || true

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
