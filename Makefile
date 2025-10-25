# Makefile for EDA Suzano project

.PHONY: help setup install ingest synthetic vecm report all clean notebook lint format test

# Default target
help:
	@echo "EDA Suzano - Available targets:"
	@echo ""
	@echo "  make setup       - Create venv and install dependencies"
	@echo "  make install     - Install dependencies only"
	@echo "  make ingest      - Load and merge data"
	@echo "  make synthetic   - Fit synthetic index and z-scores"
	@echo "  make vecm        - Fit VECM model"
	@echo "  make report      - Generate visualizations"
	@echo "  make all         - Run complete pipeline"
	@echo "  make clean       - Remove generated files"
	@echo "  make notebook    - Launch Jupyter notebook"
	@echo "  make lint        - Run linter (ruff)"
	@echo "  make format      - Format code (black)"
	@echo ""

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	python -m venv .venv
	@echo ""
	@echo "Virtual environment created!"
	@echo ""
	@echo "To activate (Windows):"
	@echo "  .venv\\Scripts\\activate"
	@echo ""
	@echo "To activate (Linux/Mac):"
	@echo "  source .venv/bin/activate"
	@echo ""
	@echo "Then run: make install"

# Install dependencies
install:
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip
	pip install -e .
	@echo "Dependencies installed!"

# Data ingestion
ingest:
	@echo "Running data ingestion..."
	python -m eda.cli ingest

# Fit synthetic index
synthetic:
	@echo "Fitting synthetic index..."
	python -m eda.cli synthetic

# Fit VECM
vecm:
	@echo "Fitting VECM..."
	python -m eda.cli vecm

# Generate report
report:
	@echo "Generating report..."
	python -m eda.cli report

# Run complete pipeline
all:
	@echo "Running complete pipeline..."
	python -m eda.cli all

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	python -m eda.cli clean
	@echo "Clean completed!"

# Launch Jupyter notebook
notebook:
	@echo "Launching Jupyter notebook..."
	jupyter notebook notebooks/EDA.ipynb

# Lint code
lint:
	@echo "Running linter..."
	ruff check src/

# Format code
format:
	@echo "Formatting code..."
	black src/
	@echo "Code formatted!"

# Run tests (placeholder)
test:
	@echo "Running tests..."
	pytest tests/ -v

