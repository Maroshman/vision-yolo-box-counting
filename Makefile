# Makefile for YOLO Box Counting Engine

.PHONY: help install setup clean test run-app run-api run-api-dev run-examples train deploy docker-build docker-run

# Default target
help:
	@echo "YOLO Box Counting Engine - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install      - Install Python dependencies"
	@echo "  setup        - Full project setup (install + create directories)"
	@echo "  clean        - Clean generated files and cache"
	@echo ""
	@echo "Running Applications:"
	@echo "  run-app      - Start Streamlit web application"
	@echo "  run-api      - Start FastAPI server (port 8000)"
	@echo "  run-api-dev  - Start FastAPI server in development mode"
	@echo "  run-examples - Run example scripts"
	@echo "  run-cli      - Show CLI usage examples"
	@echo ""
	@echo "Model Operations:"
	@echo "  train        - Train model with sample dataset"
	@echo "  evaluate     - Evaluate trained model"
	@echo "  detect       - Run detection on sample images"
	@echo ""
	@echo "Development:"
	@echo "  test         - Run tests"
	@echo "  format       - Format code with black"
	@echo "  lint         - Run code linting"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-cloud     - Deploy to Google Cloud Run"
	@echo "  test-deployment  - Test deployed API (URL=https://...)"
	@echo "  docker-build-api - Build Docker image for API"
	@echo "  docker-run-api   - Run API in Docker container"

# Installation and Setup
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

setup: install
	@echo "🔧 Setting up project structure..."
	mkdir -p data/images data/annotations results models datasets
	cp .env.example .env
	@echo "✅ Project setup complete!"
	@echo "💡 Edit .env file with your Roboflow API key"

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__ src/__pycache__ .pytest_cache
	rm -rf runs results/temp datasets/temp
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

# Running Applications
run-app:
	@echo "🚀 Starting Streamlit web application..."
	streamlit run app.py

run-api:
	@echo "🚀 Starting FastAPI server (production)..."
	python run_api.py

run-api-dev:
	@echo "🚀 Starting FastAPI server (development with hot reload)..."
	python run_api_dev.py

run-examples:
	@echo "📚 Running example scripts..."
	python examples.py --example all

run-cli:
	@echo "💻 CLI Usage Examples:"
	@echo ""
	@echo "Detect boxes in single image:"
	@echo "  python cli.py detect image.jpg -o results"
	@echo ""
	@echo "Count boxes in directory:"
	@echo "  python cli.py count data/images --format json"
	@echo ""
	@echo "Train new model:"
	@echo "  python cli.py train datasets/data.yaml --epochs 100"
	@echo ""
	@echo "Evaluate model:"
	@echo "  python cli.py evaluate models/best.pt datasets/data.yaml"

# Model Operations
train:
	@echo "🏋️ Training model with sample data..."
	@if [ -f "datasets/prepared/data.yaml" ]; then \
		python cli.py train datasets/prepared/data.yaml --epochs 50 --batch-size 16; \
	else \
		echo "❌ No dataset found. Run setup and download dataset first."; \
	fi

evaluate:
	@echo "📊 Evaluating model performance..."
	@if [ -f "runs/detect/train/weights/best.pt" ]; then \
		python cli.py evaluate runs/detect/train/weights/best.pt datasets/prepared/data.yaml; \
	else \
		echo "❌ No trained model found. Train a model first."; \
	fi

detect:
	@echo "🔍 Running detection on sample images..."
	@if [ -d "data/images" ] && [ "$(shell ls -A data/images)" ]; then \
		python cli.py detect data/images -o results/detection; \
	else \
		echo "❌ No sample images found in data/images/"; \
		echo "💡 Add some images to data/images/ directory"; \
	fi

# Development
test:
	@echo "🧪 Running tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		pytest tests/ -v; \
	else \
		echo "⚠️ pytest not installed. Installing..."; \
		pip install pytest; \
		pytest tests/ -v; \
	fi

format:
	@echo "🎨 Formatting code with black..."
	@if command -v black >/dev/null 2>&1; then \
		black src/ app.py cli.py examples.py; \
	else \
		echo "⚠️ black not installed. Installing..."; \
		pip install black; \
		black src/ app.py cli.py examples.py; \
	fi

lint:
	@echo "🔍 Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ app.py cli.py examples.py --max-line-length=100; \
	else \
		echo "⚠️ flake8 not installed. Installing..."; \
		pip install flake8; \
		flake8 src/ app.py cli.py examples.py --max-line-length=100; \
	fi

# Deployment
deploy-cloud:
	@echo "� Deploying to Google Cloud Run..."
	./deploy.sh

test-deployment:
	@echo "🧪 Testing deployed API..."
	@if [ -z "$(URL)" ]; then \
		echo "❌ Please provide URL: make test-deployment URL=https://your-service-url.run.app"; \
	else \
		python test_deployment.py $(URL); \
	fi

docker-build-api:
	@echo "🐳 Building API Docker image..."
	docker build -f Dockerfile.api -t yolo-box-counting-api:latest .

docker-run-api:
	@echo "🐳 Running API Docker container..."
	docker run -p 8080:8080 --env-file .env.cloud yolo-box-counting-api:latest

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t yolo-box-counting .
	@echo "✅ Docker image built!"

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8501:8501 -v $(PWD)/data:/app/data yolo-box-counting

# Jupyter notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter lab notebooks/

# Download sample data (if available)
download-sample:
	@echo "📥 Downloading sample data..."
	@echo "💡 Add your own sample data to data/images/"
	@echo "💡 Or configure Roboflow integration to download datasets"

# Performance benchmarking
benchmark:
	@echo "⚡ Running performance benchmark..."
	python -c "
import time
import cv2
import numpy as np
from src.box_detector import BoxDetector

detector = BoxDetector('yolov8n.pt')
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(5):
    detector.detect_boxes(dummy_image)

# Benchmark
times = []
for _ in range(20):
    start = time.time()
    detector.detect_boxes(dummy_image)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
fps = 1 / avg_time
print(f'Average inference time: {avg_time:.3f}s')
print(f'Approximate FPS: {fps:.1f}')
"

# System info
info:
	@echo "🔧 System Information:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "GPU available: $(shell python -c 'import torch; print(torch.cuda.is_available())')"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "GPU info:"; \
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
	fi
	@echo "Disk space:"
	@df -h . | tail -1