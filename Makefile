# Makefile for Video Integrity SDK

.PHONY: help install test lint format clean demo docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  test-fast   - Run fast tests only"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean temporary files"
	@echo "  demo        - Run complete demo"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-unit:
	pytest tests/test_config.py tests/test_utils.py tests/test_keys.py tests/test_models.py -v

test-integration:
	pytest tests/test_integration.py tests/test_api.py -v

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	ruff check src/ tests/ --fix

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf out/
	rm -f *.mp4 *.csv *.png
	rm -f video_secrets.db

demo: clean
	@echo "🎬 Creating demo video..."
	ffmpeg -f lavfi -i testsrc=duration=3:size=256x256:rate=10 demo_input.mp4 -y -hide_banner -loglevel error
	@echo "🔐 Embedding watermark..."
	python src/encoder.py embed demo_input.mp4 demo_watermarked.mp4 --secret ffeeddccbbaa99887766554433221100 --crf 18
	@echo "📼 Re-encoding (simulating compression attack)..."
	ffmpeg -i demo_watermarked.mp4 -c:v libx264 -crf 23 demo_compressed.mp4 -y -hide_banner -loglevel error
	@echo "🔍 Verifying watermark..."
	python src/cli.py verify demo_compressed.mp4 --secret ffeeddccbbaa99887766554433221100 --export-artifacts --summary
	@echo "📊 Demo complete! Check out/ber.png for BER timeline"
	@if command -v open >/dev/null 2>&1; then open out/ber.png; fi

docker-build:
	docker build -t video-integrity-sdk .

docker-run:
	docker run -p 8000:8000 video-integrity-sdk

# Development helpers
dev-setup: install
	pip install pre-commit
	pre-commit install

check: lint test-fast
	@echo "✅ All checks passed!"

ci: lint test
	@echo "✅ CI pipeline completed!"