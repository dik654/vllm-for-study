.PHONY: help build build-release test lint format clean docker-up docker-down migrate db-upgrade db-create run-router run-provider check

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

build:  ## Build all binaries in debug mode
	cargo build --workspace

build-release:  ## Build all binaries in release mode
	cargo build --workspace --release

test:  ## Run tests
	cargo test --workspace

check:  ## Check code without building
	cargo check --workspace

lint:  ## Run linting with clippy
	cargo clippy --workspace -- -D warnings

format:  ## Format code
	cargo fmt --all

format-check:  ## Check code formatting
	cargo fmt --all -- --check

clean:  ## Clean up build artifacts
	cargo clean
	rm -rf target/

docker-up:  ## Start all services with Docker Compose
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## View logs
	docker-compose logs -f

docker-build:  ## Build Docker images
	docker-compose build

docker-rebuild:  ## Rebuild Docker images without cache
	docker-compose build --no-cache

migrate:  ## Create new migration file
	@echo "Creating migration: $(MSG)"
	@mkdir -p migrations
	@echo "-- Migration: $(MSG)" > migrations/$(shell date +%Y%m%d%H%M%S)_$(MSG).sql
	@echo "-- Add your SQL here" >> migrations/$(shell date +%Y%m%d%H%M%S)_$(MSG).sql

db-create:  ## Create database
	docker-compose exec postgres psql -U postgres -c "CREATE DATABASE distributed_llm;"

db-upgrade:  ## Apply migrations using sqlx
	cargo install sqlx-cli --no-default-features --features postgres
	sqlx migrate run --database-url "${DATABASE_URL}"

db-reset:  ## Reset database (WARNING: destroys data)
	docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS distributed_llm;"
	docker-compose exec postgres psql -U postgres -c "CREATE DATABASE distributed_llm;"
	sqlx migrate run --database-url "${DATABASE_URL}"

run-router:  ## Run router locally
	cargo run --bin router

run-provider:  ## Run provider locally
	cargo run --bin provider

watch-router:  ## Run router with auto-reload
	cargo watch -x "run --bin router"

watch-provider:  ## Run provider with auto-reload
	cargo watch -x "run --bin provider"

dev:  ## Start development environment
	docker-compose up -d postgres redis
	@echo "Waiting for services to be ready..."
	@sleep 3
	cargo run --bin router

.DEFAULT_GOAL := help
