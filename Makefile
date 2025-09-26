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


## Initiate project (create venv, install dependencies, get data)
.PHONY: init
init: venv requirements get_llama_model
	@echo "Project initialized! 🚀✨"


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
	@$(VENV_BIN)/python -m ruff format --check
	@$(VENV_BIN)/python -m ruff check


## Format source code with ruff
.PHONY: format
format:
	@$(VENV_BIN)/python -m ruff check --fix
	@$(VENV_BIN)/python -m ruff format


## Run tests
.PHONY: test
test:
	@$(VENV_BIN)/python -m pytest tests/ -v


## Get llama model from Hugging Face
.PHONY: get_llama_model
get_llama_model:
	@echo "Downloading Llama model..."
	@. $(VENV_BIN)/activate && \
	hf download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ./models


## Create Chroma vector database (use TYPE=books|azure, default shows help)
.PHONY: chroma
chroma:
	@if [ -z "$(TYPE)" ]; then \
		echo "Chroma Database Creation"; \
		echo "========================"; \
		echo ""; \
		echo "Usage: make chroma TYPE=<database_type> [OPTIONS]"; \
		echo ""; \
		echo "Required:"; \
		echo "  TYPE=books|azure      Database type to create"; \
		echo ""; \
		echo "Optional:"; \
		echo "  CHUNK_SIZE=N          Size of text chunks (default: 300)"; \
		echo "  CHUNK_OVERLAP=N       Overlap between chunks (default: 100)"; \
		echo "  MODEL_NAME=model      HuggingFace model name"; \
		echo "  VERBOSE=1             Enable verbose output"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make chroma TYPE=books"; \
		echo "  make chroma TYPE=azure VERBOSE=1"; \
		echo "  make chroma TYPE=books CHUNK_SIZE=500"; \
	else \
		echo "Creating $(TYPE) Chroma database..."; \
		$(VENV_BIN)/python rag_project/create_chroma_database.py $(TYPE) \
			$(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE)) \
			$(if $(CHUNK_OVERLAP),--chunk-overlap $(CHUNK_OVERLAP)) \
			$(if $(MODEL_NAME),--model-name $(MODEL_NAME)) \
			$(if $(VERBOSE),--verbose); \
	fi


## Get data from the data repository
.PHONY: get_azure_data
get_azure_data:
	@$(PYTHON_INTERPRETER) rag_project/azure_get_data.py


## Fix Pip
.PHONY: fix_pip
fix_pip:
	@curl -sS https://bootstrap.pypa.io/get-pip.py | $(VENV_BIN)/python && \
	$(VENV_BIN)/python -m pip install --upgrade pip setuptools wheel


## Streamlit
.PHONY: streamlit
streamlit:
	@$(VENV_BIN)/python -m streamlit run rag_project/app.py --server.port 8501 --server.address localhost
