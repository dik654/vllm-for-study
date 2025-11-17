.PHONY: help install dev-install test lint format clean docker-up docker-down migrate db-upgrade db-downgrade

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

dev-install:  ## Install development dependencies
	poetry install

test:  ## Run tests
	pytest -v --cov=router --cov=provider --cov=shared

lint:  ## Run linting
	black --check .
	isort --check-only .
	flake8 .
	mypy router provider shared

format:  ## Format code
	black .
	isort .

clean:  ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage

docker-up:  ## Start all services with Docker Compose
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## View logs
	docker-compose logs -f

docker-build:  ## Build Docker images
	docker-compose build

migrate:  ## Create new migration
	alembic revision --autogenerate -m "$(MSG)"

db-upgrade:  ## Apply migrations
	alembic upgrade head

db-downgrade:  ## Rollback last migration
	alembic downgrade -1

db-reset:  ## Reset database (WARNING: destroys data)
	alembic downgrade base
	alembic upgrade head

run-router:  ## Run router locally
	python -m router.main

run-provider:  ## Run provider locally
	python -m provider.main

.DEFAULT_GOAL := help
