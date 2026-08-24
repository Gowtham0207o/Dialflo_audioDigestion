.PHONY: help install dev run test test-unit test-integration test-smoke lint format typecheck clean docker-build docker-up docker-down eval

# ── Variables ──────────────────────────────
PYTHON := python
PIP := pip
PYTEST := pytest
APP_MODULE := app.main:create_app

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install all dependencies (dev + eval)
	$(PIP) install -e ".[dev,eval]"
	pre-commit install

# ── Run ────────────────────────────────────
run: ## Run the development server
	uvicorn $(APP_MODULE) --factory --reload --host 0.0.0.0 --port 8000

# ── Test ───────────────────────────────────
test: ## Run all tests
	$(PYTEST) -v --cov=src/app --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v -m integration

test-smoke: ## Run smoke test against running Docker service
	./scripts/smoke_test.sh

# ── Code Quality ───────────────────────────
lint: ## Run linter
	ruff check src/ tests/ eval/

format: ## Format code
	ruff format src/ tests/ eval/

typecheck: ## Run type checker
	mypy src/

# ── Docker ─────────────────────────────────
docker-build: ## Build Docker image
	docker compose build

docker-up: ## Start service via Docker Compose
	docker compose up -d

docker-down: ## Stop Docker Compose services
	docker compose down

# ── Eval ───────────────────────────────────
eval: ## Run evaluation harness
	$(PYTHON) -m eval.run_eval

# ── Cleanup ────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build *.egg-info htmlcov .coverage
