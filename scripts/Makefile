# Makefile for Wiener Index Analysis - Modular Architecture

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

# Test the modular architecture
.PHONY: test
test: setup
	@echo "Running quick test (modular architecture)..."
	$(PYTHON) $(MAIN_SCRIPT) --quick-test

# Test individual modules
.PHONY: test-modules
test-modules: setup
	@echo "Testing individual modular components..."
	@$(PYTHON) -c "\
print('Testing modular architecture components...'); \
from core.point import Point; \
from generators.point_generator import PointGenerator; \
from solvers.brute_force_solver import BruteForceSolver; \
from core.wiener_index_calculator import WienerIndexCalculator; \
print('✅ Core imports successful'); \
gen = PointGenerator(); \
points = gen.generate_convex_hull_points(4, seed=42); \
print(f'✅ Point generation: {len(points)} points'); \
calc = WienerIndexCalculator(); \
solver = BruteForceSolver(); \
print('✅ All modules loaded successfully'); \
print('🎯 Modular architecture test: SUCCESS!')"

# Run the modular Wiener index analysis
.PHONY: run
run: setup
	@echo "Running Wiener index analysis (modular architecture)..."
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
	@echo "Available targets for Wiener Index Analysis (Modular Architecture):"
	@echo "========================================================================"
	@echo "  all           - Setup environment and run analysis"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  install       - Install/update requirements in existing venv"
	@echo "  run           - Run the complete Wiener index analysis"
	@echo "  test          - Run quick test with minimal parameters"
	@echo "  test-modules  - Test individual modular components"
	@echo "  custom        - Run with custom parameters (use ARGS='...')"
	@echo "  clean         - Remove generated files and results"
	@echo "  clean-all     - Remove everything including virtual environment"
	@echo "  info          - Show project information"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup         # Set up the environment"
	@echo "  make test-modules  # Test modular components"
	@echo "  make test          # Quick test run"
	@echo "  make run           # Complete analysis"
	@echo ""
	@echo "Custom Usage Examples:"
	@echo "  make custom ARGS='--quick-test'"
	@echo "  make custom ARGS='--point-counts 4 5 6 --trials 3'"
	@echo "  make custom ARGS='--output-dir my_results --point-types convex'"
	@echo ""
	@echo "Modular Architecture:"
	@echo "  📦 core/          - Point class and Wiener index calculation"
	@echo "  📦 generators/    - Point set generation (convex hull, general)"
	@echo "  📦 solvers/       - Algorithm implementations (brute force, D&C)"
	@echo "  📦 analysis/      - Statistical analysis and performance comparison"
	@echo "  📦 visualization/ - Plotting and visualization tools"
	@echo "  📦 utils/         - Data management and logging utilities"
	@echo ""
	@echo "Benefits of modular design:"
	@echo "  • Easy to test individual components"
	@echo "  • Clean separation of concerns"
	@echo "  • Extensible for new algorithms/features"
	@echo "  • Maintainable codebase"
