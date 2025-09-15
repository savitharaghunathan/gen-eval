.PHONY: help install dev pre-commit pre-push test lint format build clean

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the package
	uv sync --dev --all-extras

dev:  ## Install pre-commit hooks
	uv run pre-commit install

pre-commit:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

pre-push:  ## Run clean, format and lint before pushing
	$(MAKE) clean
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

test:  ## Run tests
	uv run pytest tests/ -v --cov=geneval --cov-report=xml --cov-report=html

lint:  ## Run linters
	uv run black --check --diff .
	uv run isort --check-only --diff .
	uv run ruff check geneval/ tests/


format:  ## Format code
	uv run black .
	uv run isort .
	uv run ruff check --fix geneval/ tests/

build:  ## Build the package
	uv build

clean:  ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
