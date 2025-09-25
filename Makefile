#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = rag_project
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
AZURE_DATA_REPO = https://github.com/MicrosoftDocs/azure-docs.git
AZURE_DATA_DIR = data/azure-docs
VENV_NAME = venv
VENV_BIN = $(VENV_NAME)/bin

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Create Python virtual environment
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	@$(PYTHON_INTERPRETER)$(PYTHON_VERSION) -m venv $(VENV_NAME)
	@echo "Virtual environment created! 🐍✨"

## Install Python dependencies
.PHONY: requirements
requirements:
	@echo "Installing Python dependencies..."
	@$(VENV_BIN)/python -m ensurepip --upgrade
	@$(VENV_BIN)/python -m pip install --upgrade pip setuptools wheel
	@$(VENV_BIN)/pip install -r requirements.txt
	@echo "Dependencies installed! 📦✨"


## Delete all compiled Python files and caches
.PHONY: clean
clean:
	@echo "Cleaning Python cache files..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@echo "Cleaning test cache files..."
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cleaning build and distribution files..."
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "build" -exec rm -rf {} +
	@find . -type d -name "dist" -exec rm -rf {} +
	@find . -type d -name ".cache" -exec rm -rf {} +
	@echo "Cleaning Jupyter notebook cache..."
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Clean complete! 🧹✨"


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	@ruff format --check
	@ruff check


## Format source code with ruff
.PHONY: format
format:
	@ruff check --fix
	@ruff format


## Run tests
.PHONY: test
test:
	@python -m pytest tests


## Get llama model from Hugging Face
.PHONY: get_llama_model
get_llama_model:
	@hf download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ./models


## Get data from the data repository
.PHONY: get_azure_data
get_azure_data:
	@$(PYTHON_INTERPRETER) rag_project/azure_get_data.py


## Fix Pip
.PHONY: fix_pip
fix_pip:
	curl -sS https://bootstrap.pypa.io/get-pip.py | python


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
