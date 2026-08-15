.PHONY: help setup all build up down restart ps logs logs-api logs-qdrant logs-mlflow clean ui-api ui-qdrant ui-mlflow ui-minio ui-n8n ui-all test

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Rag Playground - Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

setup: ## Initial setup (create .env from .env.example if missing)
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env from .env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✓ .env created! Please review and update values.$(NC)"; \
	else \
		echo "$(GREEN)✓ .env already exists$(NC)"; \
	fi
	@mkdir -p data/documents data/golden_set
	@touch data/documents/.gitkeep data/golden_set/.gitkeep
	@echo "$(GREEN)✓ Data directories created$(NC)"

all: setup build up ## Setup + Build + Start all services
	@echo "$(GREEN)✓ All services started!$(NC)"
	@echo "$(BLUE)Run 'make ui-all' to open all UIs$(NC)"

build: ## Build Docker images
	docker compose build

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

restart: down up ## Restart all services

ps: ## Show service status
	docker compose ps

logs: ## Show logs for all services
	docker compose logs -f

logs-api: ## Show API logs
	docker compose logs -f api

logs-qdrant: ## Show Qdrant logs
	docker compose logs -f qdrant

logs-mlflow: ## Show MLflow logs
	docker compose logs -f mlflow

clean: down ## Stop services and remove volumes
	docker compose down -v
	@echo "$(YELLOW)⚠ All data volumes removed!$(NC)"

# UI shortcuts
ui-api: ## Open API docs (Swagger)
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs

ui-qdrant: ## Open Qdrant dashboard
	@open http://localhost:6333/dashboard || xdg-open http://localhost:6333/dashboard

ui-mlflow: ## Open MLflow UI
	@open http://localhost:5001 || xdg-open http://localhost:5001

ui-minio: ## Open MinIO console
	@open http://localhost:9001 || xdg-open http://localhost:9001

ui-n8n: ## Open n8n workflow editor
	@open http://localhost:5678 || xdg-open http://localhost:5678

ui-all: ui-api ui-qdrant ui-mlflow ui-minio ui-n8n ## Open all UIs

# Health checks
test: ## Test all services
	@echo "$(BLUE)Testing LM Studio...$(NC)"
	@curl -s http://localhost:1234/v1/models > /dev/null && echo "$(GREEN)✓ LM Studio OK$(NC)" || echo "$(YELLOW)✗ LM Studio not running$(NC)"
	@echo "$(BLUE)Testing Ollama...$(NC)"
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "$(GREEN)✓ Ollama OK$(NC)" || echo "$(YELLOW)✗ Ollama not running$(NC)"
	@echo "$(BLUE)Testing API...$(NC)"
	@curl -s http://localhost:8000/health > /dev/null && echo "$(GREEN)✓ API OK$(NC)" || echo "$(YELLOW)✗ API not running$(NC)"
	@echo "$(BLUE)Testing Qdrant...$(NC)"
	@curl -s http://localhost:6333/readyz > /dev/null && echo "$(GREEN)✓ Qdrant OK$(NC)" || echo "$(YELLOW)✗ Qdrant not running$(NC)"