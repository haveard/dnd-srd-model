# D&D SRD LoRA Fine-Tuning Pipeline Makefile
# For Apple Silicon Mac (M4 processor)

PYTHON := python3
PIP := pip3
STREAMLIT := streamlit
UVICORN := uvicorn

# Directories
PROJECT_DIR := $(PROJECT_DIR)
DATA_DIR := $(PROJECT_DIR)/data
MODELS_DIR := $(PROJECT_DIR)/models
EVAL_DIR := $(PROJECT_DIR)/eval
SCRIPTS_DIR := $(PROJECT_DIR)/scripts

# Files
REQUIREMENTS := $(PROJECT_DIR)/requirements.txt
DATASET_FILE := $(DATA_DIR)/dnd_srd_qa.jsonl
PYTHIA_CHECKPOINT := $(MODELS_DIR)/pythia-lora-checkpoint
DISTILGPT2_CHECKPOINT := $(MODELS_DIR)/distilgpt2-dnd-lora
EVAL_RESULTS := $(EVAL_DIR)/eval_scores.csv

.PHONY: help setup install-deps prepare-data train-pythia train-distilgpt2 train-all compare-pythia compare-distilgpt2 compare-all serve-api demo dashboard test-models clean all

help: ## Show this help message
	@echo "D&D SRD LoRA Fine-Tuning Pipeline"
	@echo "=================================="
	@echo "🎯 COMPLETE PROJECT - Zero to Hero D&D Knowledge Injection"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make all              # Complete pipeline"
	@echo "  make demo             # Show dramatic improvements"
	@echo "  make serve-api        # Start comparison API"
	@echo "  make dashboard        # Launch interactive UI"

setup: ## Create project directories
	@echo "🏗️  Setting up project structure..."
	@mkdir -p $(DATA_DIR) $(MODELS_DIR) $(EVAL_DIR) $(SCRIPTS_DIR)
	@echo "✅ Project structure created"

install-deps: $(REQUIREMENTS) ## Install Python dependencies
	@echo "📦 Installing Python dependencies..."
	@cd $(PROJECT_DIR) && $(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

prepare-data: $(DATASET_FILE) ## Prepare training dataset from D&D SRD
$(DATASET_FILE): $(SCRIPTS_DIR)/prepare_dataset.py
	@echo "📚 Preparing D&D SRD dataset..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/prepare_dataset.py
	@echo "✅ Dataset prepared: $(DATASET_FILE)"

train-pythia: $(PYTHIA_CHECKPOINT) ## Train Pythia-1.4B LoRA model
$(PYTHIA_CHECKPOINT): $(DATASET_FILE) $(SCRIPTS_DIR)/train_lora_pythia_fixed.py
	@echo "🧠 Training Pythia-1.4B LoRA model..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/train_lora_pythia_fixed.py
	@echo "✅ Pythia model training completed"

train-distilgpt2: $(DISTILGPT2_CHECKPOINT) ## Train DistilGPT2 LoRA model  
$(DISTILGPT2_CHECKPOINT): $(DATASET_FILE) $(SCRIPTS_DIR)/train_distilgpt2_dnd.py
	@echo "🧠 Training DistilGPT2 LoRA model..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/train_distilgpt2_dnd.py
	@echo "✅ DistilGPT2 model training completed"

train-all: train-pythia train-distilgpt2 ## Train both models

compare-pythia: $(PYTHIA_CHECKPOINT) ## Compare original vs Pythia LoRA
	@echo "🔍 Comparing Original vs Pythia LoRA..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/compare_original_vs_lora.py
	@echo "✅ Pythia comparison completed"

compare-distilgpt2: $(DISTILGPT2_CHECKPOINT) ## Compare original vs DistilGPT2 LoRA
	@echo "🔍 Comparing Original vs DistilGPT2 LoRA..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/compare_distilgpt2_dnd.py
	@echo "✅ DistilGPT2 comparison completed"

compare-all: compare-pythia compare-distilgpt2 ## Run all comparisons

demo: $(DISTILGPT2_CHECKPOINT) ## Show dramatic improvement demonstration
	@echo "🎲 Demonstrating Zero-to-Hero Transformation..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/demo_dramatic_improvement.py
	@echo "✅ Demonstration completed"

serve-api: $(DISTILGPT2_CHECKPOINT) ## Start FastAPI server for model comparison
	@echo "🚀 Starting API server at http://localhost:8000..."
	@echo "📖 API docs available at http://localhost:8000/docs"
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/api_server.py

serve-api: $(MODEL_CHECKPOINT) ## Start the FastAPI inference server
	@echo "🚀 Starting FastAPI inference server..."
	@echo "   Server will be available at: http://localhost:8000"
	@echo "   Press Ctrl+C to stop the server"
	@cd $(PROJECT_DIR) && $(UVICORN) scripts.inference_server_lora:app --reload --host 0.0.0.0 --port 8000

serve-test-api: ## Start the test API server
	@echo "🧪 Starting test FastAPI inference server..."
	@echo "   Server will be available at: http://localhost:8001"
	@echo "   Press Ctrl+C to stop the server"
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/test_api_server.py

quick-test: ## Run quick test training with small model
	@echo "🧪 Starting quick test training..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/quick_test_training.py
	@echo "✅ Quick test completed"

serve-api-bg: $(MODEL_CHECKPOINT) ## Start API server in background
	@echo "🚀 Starting FastAPI server in background..."
	@cd $(PROJECT_DIR) && nohup $(UVICORN) scripts.inference_server_lora:app --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &
	@echo "✅ API server started in background (PID: $$!)"
	@echo "   Logs: $(PROJECT_DIR)/api_server.log"
	@echo "   Server: http://localhost:8000"

stop-api: ## Stop background API server
	@echo "🛑 Stopping API server..."
	@pkill -f "inference_server_lora" || true
	@echo "✅ API server stopped"

evaluate: $(EVAL_RESULTS) ## Evaluate model against GPT-4
$(EVAL_RESULTS): $(MODEL_CHECKPOINT) $(SCRIPTS_DIR)/compare_lora_vs_gpt4_eval.py
	@echo "📊 Starting model evaluation..."
	@echo "   Note: Requires OPENAI_API_KEY environment variable for GPT-4 comparison"
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "⚠️  Warning: OPENAI_API_KEY not set. GPT-4 comparison will be skipped."; \
	fi
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/compare_lora_vs_gpt4_eval.py
	@echo "✅ Evaluation completed: $(EVAL_RESULTS)"

dashboard: ## Launch Streamlit user interface
	@echo "🖥️  Starting Streamlit user interface..."
	@echo "   Dashboard will be available at: http://localhost:8501"
	@echo "   Press Ctrl+C to stop the dashboard"
	@cd $(PROJECT_DIR) && $(STREAMLIT) run streamlit_app.py

eval-dashboard: ## Launch evaluation dashboard
	@echo "📊 Starting evaluation dashboard..."
	@echo "   Dashboard will be available at: http://localhost:8502"
	@echo "   Press Ctrl+C to stop the dashboard"
	@cd $(PROJECT_DIR) && $(STREAMLIT) run streamlit_eval_dashboard.py --server.port 8502

check-gpu: ## Check GPU/MPS availability
	@echo "🔍 Checking compute capabilities..."
	@$(PYTHON) -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

status: ## Check status of all components
	@echo "🔍 System Status Check"
	@echo "====================="
	@echo ""
	@echo "📦 Dependencies:"
		@$(PYTHON) -c "import transformers, datasets, peft, fastapi, streamlit; print('✅ All required packages installed')" 2>/dev/null || echo "❌ Missing dependencies (run: make install-deps)"
	@echo ""
	@echo "📊 Compute:"
	@$(PYTHON) -c "import torch; print(f'✅ MPS available: {torch.backends.mps.is_available()}')" 2>/dev/null || echo "❌ PyTorch not available"
	@echo ""
	@echo "📁 Files:"
	@[ -f "$(DATASET_FILE)" ] && echo "✅ Dataset ready" || echo "❌ Dataset not found (run: make prepare-data)"
	@[ -d "$(PYTHIA_CHECKPOINT)" ] && echo "✅ Pythia model trained" || echo "❌ Pythia model not trained (run: make train-pythia)"
	@[ -d "$(DISTILGPT2_CHECKPOINT)" ] && echo "✅ DistilGPT2 model trained" || echo "❌ DistilGPT2 model not trained (run: make train-distilgpt2)"

clean: ## Clean generated files
	@echo "🧹 Cleaning generated files..."
	@rm -rf $(DATA_DIR)/*.jsonl $(DATA_DIR)/*.json
	@rm -rf $(MODELS_DIR)/*
	@rm -rf $(EVAL_DIR)/*
	@rm -f $(PROJECT_DIR)/api_server.log
	@rm -rf $(PROJECT_DIR)/__pycache__ $(PROJECT_DIR)/scripts/__pycache__
	@echo "✅ Cleanup completed"

# Complete pipeline
all: setup install-deps prepare-data train-all compare-all ## Run complete pipeline
	@echo "🎉 COMPLETE D&D LORA PIPELINE FINISHED!"
	@echo ""
	@echo "🏆 RESULTS:"
	@echo "  📊 Training: 2 models successfully trained"
	@echo "  📈 Evaluation: Comprehensive comparisons completed"
	@echo "  🎯 Transformation: Zero-to-Hero D&D knowledge injection achieved"
	@echo ""
	@echo "🚀 NEXT STEPS:"
	@echo "  make demo             # See dramatic improvements"
	@echo "  make serve-api        # Start comparison API server"
	@echo "  make dashboard        # Launch interactive UI"
	@echo ""
	@echo "📖 View results:"
	@echo "  - HTML reports in eval/ directories"
	@echo "  - Final report: FINAL_PROJECT_REPORT.md"


# Testing
test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

train-quick: ## Quick training test with smaller model
	@echo "⚡ Running quick training test..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/quick_test_training.py
	@echo "✅ Quick training test completed"

train-distilgpt2: ## Train LoRA on distilgpt2 for dramatic D&D knowledge injection demo
	@echo "🎯 Training distilgpt2 with D&D knowledge for dramatic comparison..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/train_distilgpt2_dnd.py
	@echo "✅ distilgpt2 D&D training completed"

compare-distilgpt2: ## Compare original distilgpt2 vs LoRA D&D-trained version
	@echo "🔍 Comparing original distilgpt2 vs D&D-trained version..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/compare_distilgpt2_dnd.py
	@echo "✅ distilgpt2 comparison completed"

live-demo: $(DISTILGPT2_CHECKPOINT) ## Live API-based demonstration of model improvements
	@echo "🎲 Starting live demonstration..."
	@echo "🌐 Using API server at http://localhost:8000"
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/live_demo.py

live-demo-quick: $(DISTILGPT2_CHECKPOINT) ## Quick live demonstration test
	@echo "⚡ Quick live demo test..."
	@cd $(PROJECT_DIR) && $(PYTHON) scripts/live_demo.py --mode quick
