# Makefile for Wiener Index Analysis - Standalone Implementation

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS = requirements.txt
MAIN_SCRIPT = main.py

# Default target
.PHONY: all
all: setup run

# Setup virtual environment and install dependencies
.PHONY: setup
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: $(REQUIREMENTS)
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@echo "Virtual environment setup complete!"

# Quick test run with minimal parameters
.PHONY: test
test: setup
	@echo "Running quick test..."
	$(PYTHON) $(MAIN_SCRIPT) --quick-test

# Run the standalone Wiener index analysis
.PHONY: run
run: setup
	@echo "Running Wiener index analysis..."
	$(PYTHON) $(MAIN_SCRIPT)

# Install/update requirements
.PHONY: install
install: $(VENV_DIR)/bin/activate
	@echo "Installing/updating requirements..."
	$(PIP) install -r $(REQUIREMENTS)

# Clean up generated files
.PHONY: clean
clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -f *.pyc
	rm -f *.log
	rm -rf wiener_results/
	rm -rf wiener_logs/
	@echo "Cleanup complete!"

# Clean everything including virtual environment
.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Full cleanup complete!"

# Run analysis with custom parameters
.PHONY: custom
custom: setup
	@echo "Running custom analysis..."
	@echo "Usage: make custom ARGS='--point-counts 4 5 6 --trials 3'"
	$(PYTHON) $(MAIN_SCRIPT) $(ARGS)

# Show project info
.PHONY: info
info:
	@echo "Project Information:"
	@echo "==================="
	@echo "Working Directory: $(PWD)"
	@echo "Main Script: $(MAIN_SCRIPT)"
	@echo "Virtual Environment: $(VENV_DIR)"
	@if [ -f $(VENV_DIR)/bin/python ]; then \
		echo "Python Version: $$($(PYTHON) --version)"; \
		echo "Virtual environment is set up."; \
	else \
		echo "Virtual environment not found. Run 'make setup' first."; \
	fi

# Show help
.PHONY: help
help:
	@echo "Available targets for Wiener Index Analysis (Standalone Implementation):"
	@echo "=========================================================================="
	@echo "  all           - Setup environment and run analysis"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  install       - Install/update requirements in existing venv"
	@echo "  run           - Run the complete Wiener index analysis"
	@echo "  test          - Run quick test with minimal parameters"
	@echo "  custom        - Run with custom parameters (use ARGS='...')"
	@echo "  clean         - Remove generated files and results"
	@echo "  clean-all     - Remove everything including virtual environment"
	@echo "  info          - Show project information"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup    # Set up the environment"
	@echo "  make test     # Quick test run"
	@echo "  make run      # Complete analysis"
	@echo ""
	@echo "Custom Usage Examples:"
	@echo "  make custom ARGS='--quick-test'"
	@echo "  make custom ARGS='--point-counts 4 5 6 --trials 3'"
	@echo "  make custom ARGS='--output-dir my_results'"
	@echo ""
	@echo "Features of the standalone implementation:"
	@echo "  • Point generation (convex hull, general)"
	@echo "  • Multiple algorithms (brute force, divide & conquer)"
	@echo "  • Statistical analysis and performance comparison"
	@echo "  • Automatic visualization and result export"
	@echo "  • All functionality in a single main.py file"
