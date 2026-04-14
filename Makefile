# ============================================================
# Credit Risk Classifier — Makefile
# ============================================================

.PHONY: help install data train test lint docker-build docker-run docker-stop clean all

PYTHON     := python
PIP        := pip
IMAGE_NAME := credit-risk-classifier
PORT       := 5000

# Default target
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║      Credit Risk Classifier — Makefile Commands      ║"
	@echo "╠══════════════════════════════════════════════════════╣"
	@echo "║  make install      Install Python dependencies       ║"
	@echo "║  make data         Generate synthetic dataset        ║"
	@echo "║  make train        Train the model                   ║"
	@echo "║  make test         Run all tests                     ║"
	@echo "║  make run          Start Flask dev server            ║"
	@echo "║  make docker-build Build Docker image                ║"
	@echo "║  make docker-run   Run Docker container              ║"
	@echo "║  make docker-stop  Stop Docker container             ║"
	@echo "║  make clean        Remove generated artifacts        ║"
	@echo "║  make all          install + data + train + test     ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@echo ""

# ---- Setup ----------------------------------------------------------------

install:
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# ---- Data -----------------------------------------------------------------

data:
	@echo "🔧 Generating synthetic dataset..."
	$(PYTHON) data/generate_data.py --samples 5000
	@echo "✅ Dataset ready at data/dataset.csv"

# ---- Training -------------------------------------------------------------

train:
	@echo "🚀 Training Credit Risk Classifier..."
	$(PYTHON) -m src.train --lr 0.05 --iters 1000
	@echo "✅ Training complete. Artifacts saved to models/"
	@cat models/metrics.txt

# ---- Testing --------------------------------------------------------------

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --tb=short
	@echo "✅ Tests complete"

test-coverage:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "✅ Coverage report saved to htmlcov/"

# ---- API ------------------------------------------------------------------

run:
	@echo "🌐 Starting Flask development server on port $(PORT)..."
	FLASK_DEBUG=true $(PYTHON) app.py

# ---- Docker ---------------------------------------------------------------

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(IMAGE_NAME):latest .
	@echo "✅ Image built: $(IMAGE_NAME):latest"

docker-run:
	@echo "🚀 Starting Docker container..."
	docker run -d \
		--name $(IMAGE_NAME) \
		-p $(PORT):$(PORT) \
		-e FLASK_DEBUG=false \
		$(IMAGE_NAME):latest
	@echo "✅ Container running on http://localhost:$(PORT)"
	@echo "   Health: http://localhost:$(PORT)/health"
	@echo "   Metrics: http://localhost:$(PORT)/metrics"

docker-stop:
	@echo "🛑 Stopping container..."
	docker stop $(IMAGE_NAME) && docker rm $(IMAGE_NAME)
	@echo "✅ Container stopped"

docker-logs:
	docker logs -f $(IMAGE_NAME)

# ---- Clean ----------------------------------------------------------------

clean:
	@echo "🧹 Cleaning generated artifacts..."
	rm -f models/model.pkl models/scaler.pkl models/metrics.txt
	rm -f data/dataset.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

# ---- All ------------------------------------------------------------------

all: install data train test
	@echo ""
	@echo "🎉 Full pipeline complete!"
