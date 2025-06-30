# Makefile for Wiener Index Analysis Project

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS = requirements.txt
MAIN_SCRIPT = main.py
SUMMARY_SCRIPT = summary.py
HAMILTONIAN_SCRIPT = hamiltonian_wiener.py

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

# Install/update requirements
.PHONY: install
install: $(VENV_DIR)/bin/activate
	@echo "Installing/updating requirements..."
	$(PIP) install -r $(REQUIREMENTS)

# Run the main analysis
.PHONY: run
run: setup
	@echo "Running main analysis..."
	$(PYTHON) $(HAMILTONIAN_SCRIPT)

# Run the summary script
.PHONY: summary
summary: setup
	@echo "Running summary analysis..."
	$(PYTHON) $(SUMMARY_SCRIPT)

# Run the hamiltonian analysis
.PHONY: hamiltonian
hamiltonian: setup
	@echo "Running hamiltonian analysis..."
	$(PYTHON) $(HAMILTONIAN_SCRIPT)

# Run multithreaded version
.PHONY: multithread
multithread: setup
	@echo "Running multithreaded analysis..."
	$(PYTHON) multithreaded_hamiltonian_general.py


# Clean up generated files
.PHONY: clean
clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -f *.pyc
	rm -f */*.pyc
	rm -f */*/*.pyc
	rm -f *.log
	rm -f wiener_*.txt
	rm -f quick_test_results.json
	@echo "Cleanup complete!"

# Clean everything including virtual environment
.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Full cleanup complete!"

# Reinstall everything from scratch
.PHONY: reinstall
reinstall: clean-all setup
	@echo "Reinstallation complete!"

# Create requirements.txt if it doesn't exist
.PHONY: create-requirements
create-requirements:
	@if [ ! -f $(REQUIREMENTS) ]; then \
		echo "Creating requirements.txt..."; \
		echo "numpy>=1.21.0" > $(REQUIREMENTS); \
		echo "matplotlib>=3.5.0" >> $(REQUIREMENTS); \
		echo "scipy>=1.7.0" >> $(REQUIREMENTS); \
		echo "tqdm>=4.62.0" >> $(REQUIREMENTS); \
		echo "pandas>=1.3.0" >> $(REQUIREMENTS); \
		echo "seaborn>=0.11.0" >> $(REQUIREMENTS); \
		echo "plotly>=5.0.0" >> $(REQUIREMENTS); \
		echo "Requirements.txt created with basic dependencies!"; \
	else \
		echo "Requirements.txt already exists."; \
	fi

# Show help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all           - Setup environment and run main analysis"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  install       - Install/update requirements in existing venv"
	@echo "  run           - Run main analysis (main.py)"
	@echo "  summary       - Run summary analysis (summary.py)"
	@echo "  hamiltonian   - Run hamiltonian analysis (hamiltonian_wiener.py)"
	@echo "  multithread   - Run multithreaded analysis"
	@echo "  test          - Run quick test to verify setup"
	@echo "  clean         - Remove generated files and cache"
	@echo "  clean-all     - Remove everything including virtual environment"
	@echo "  reinstall     - Clean everything and reinstall from scratch"
	@echo "  create-requirements - Create basic requirements.txt if missing"
	@echo "  help          - Show this help message"

# Advanced targets for development

# Interactive Python shell with environment loaded
.PHONY: shell
shell: setup
	@echo "Starting Python shell with virtual environment..."
	$(PYTHON)

# Install development dependencies
.PHONY: dev-install
dev-install: setup
	@echo "Installing development dependencies..."
	$(PIP) install ipython jupyter black flake8 pytest
	@echo "Development dependencies installed!"

# Format code with black
.PHONY: format
format: setup
	@echo "Formatting code with black..."
	$(PYTHON) -m black . --exclude $(VENV_DIR)

# Lint code with flake8
.PHONY: lint
lint: setup
	@echo "Linting code with flake8..."
	$(PYTHON) -m flake8 . --exclude $(VENV_DIR) --max-line-length=88

# Run Jupyter notebook
.PHONY: notebook
notebook: dev-install
	@echo "Starting Jupyter notebook..."
	$(VENV_DIR)/bin/jupyter notebook

# Show virtual environment info
.PHONY: info
info:
	@echo "Project Information:"
	@echo "==================="
	@echo "Virtual Environment: $(VENV_DIR)"
	@if [ -f $(VENV_DIR)/bin/python ]; then \
		echo "Python Version: $$($(PYTHON) --version)"; \
		echo "Pip Version: $$($(PIP) --version)"; \
		echo "Installed Packages:"; \
		$(PIP) list; \
	else \
		echo "Virtual environment not found. Run 'make setup' first."; \
	fi

# Export current environment to requirements.txt
.PHONY: freeze
freeze: setup
	@echo "Exporting current environment to requirements.txt..."
	$(PIP) freeze > $(REQUIREMENTS)
	@echo "Requirements exported to $(REQUIREMENTS)"

# Backup important files
.PHONY: backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup_$$(date +%Y%m%d_%H%M%S).tar.gz \
		--exclude=venv \
		--exclude=__pycache__ \
		--exclude=*.pyc \
		--exclude=*.log \
		--exclude=backups \
		.
	@echo "Backup created in backups/ directory"
