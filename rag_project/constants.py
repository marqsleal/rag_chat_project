import os
from pathlib import Path

ROOT_DIR: str = Path(__file__).parent.parent.absolute()
"""Root directory of the project."""

SENTENCE_TRANSFORMERS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
"""Model name for generating embeddings using HuggingFaceEmbeddings."""

# Common directories
LOG_DIR: str = os.path.join(ROOT_DIR, "logs")
"""Directory to store all log files."""
DATA_DIR: str = os.path.join(ROOT_DIR, "data")
"""Directory to store all data files."""
MODELS_DIR: str = os.path.join(ROOT_DIR, "models")
"""Directory to store all model files."""
CONFIG_DIR: str = os.path.join(ROOT_DIR, "config")
"""Directory to store all configuration files."""
LOGS_DIR: str = os.path.join(ROOT_DIR, "logs")
"""Directory to store all log files."""
DOCS_DIR: str = os.path.join(ROOT_DIR, "docs")
"""Directory to store all documentation files."""
TESTS_DIR: str = os.path.join(ROOT_DIR, "tests")
"""Directory to store all test files."""

LLAMA_MODEL_NAME: str = "llama-2-7b-chat.Q4_K_M.gguf"
"""LLaMA model file name."""
LLAMA_MODEL_PATH: str = os.path.join(MODELS_DIR, LLAMA_MODEL_NAME)
"""Path to the LLaMA model file."""

CHROMA_DB_DIR: str = os.path.join(DATA_DIR, "chroma-db")
"""Directory to store Chroma vector database files."""

# Azure documentation constants
AZURE_COLLECTION_NAME: str = "azure-docs"
"""Chroma collection name for Azure documentation."""
AZURE_RAW_DATA_DIR: str = os.path.join(DATA_DIR, AZURE_COLLECTION_NAME)
"""Directory to store raw Azure documentation data in .md format."""
AZURE_CHROMA_DB_DIR: str = os.path.join(CHROMA_DB_DIR, AZURE_COLLECTION_NAME)
"""Directory to store Chroma vector database files."""
AZURE_GIT_URL: str = "https://github.com/MicrosoftDocs/azure-docs.git"
"""Git repository URL for Azure documentation."""

# Books constants
BOOKS_COLLECTION_NAME: str = "books"
"""Chroma collection name for books."""
BOOKS_RAW_DATA_DIR: str = os.path.join(DATA_DIR, BOOKS_COLLECTION_NAME)
"""Directory to store raw books data in .md format."""
BOOKS_CHROMA_DB_DIR: str = os.path.join(CHROMA_DB_DIR, BOOKS_COLLECTION_NAME)
"""Directory to store Chroma vector database files."""
