import argparse
import os
import shutil
import sys

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag_project.constants import (
    AZURE_CHROMA_DB_DIR,
    AZURE_COLLECTION_NAME,
    AZURE_RAW_DATA_DIR,
    BOOKS_CHROMA_DB_DIR,
    BOOKS_COLLECTION_NAME,
    BOOKS_RAW_DATA_DIR,
    SENTENCE_TRANSFORMERS_MODEL_NAME,
)
from rag_project.logger import logger

DEFAULT_CHUNK_SIZE: int = 300
"""Default size of text chunks for text splitting."""
DEFAULT_CHUNK_OVERLAP: int = 100
"""Default overlap between text chunks for text splitting."""
EXAMPLE_DOCUMENT_INDEX: int = 10
"""Index of the example document to log after splitting."""
CONTENT_PREVIEW_LENGTH: int = 100
"""Number of characters to show in content preview for logging."""
MARKDOWN_GLOB_PATTERN: str = "**/*.md"
"""Glob pattern to match markdown files in the directory."""


def load_documents(directory: str) -> list[Document]:
    """Load documents from a directory with error handling."""
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")

        logger.info("Loading documents from %s...", directory)
        loader = DirectoryLoader(directory, glob=MARKDOWN_GLOB_PATTERN)
        documents = loader.load()

        if not documents:
            logger.warning("No documents found in %s", directory)
            return []

        logger.info("Loaded %d documents from %s.", len(documents), directory)
        return documents
    except Exception as e:
        logger.error("Failed to load documents from %s: %s", directory, str(e))
        raise


def split_text(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks with configurable parameters."""
    if not documents:
        logger.warning("No documents to split")
        return []

    logger.info("Splitting text from %d documents...", len(documents))

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))

        if chunks and len(chunks) > EXAMPLE_DOCUMENT_INDEX:
            document = chunks[EXAMPLE_DOCUMENT_INDEX]
            logger.info(
                "Document content example: %s",
                document.page_content[:CONTENT_PREVIEW_LENGTH] + "...",
            )
            logger.info("Document metadata example: %s", document.metadata)

        return chunks
    except Exception as e:
        logger.error("Failed to split documents: %s", str(e))
        raise


def clean_chroma_dir(path: str) -> None:
    """Clean Chroma directory."""
    try:
        if os.path.exists(path):
            logger.info("Removing existing directory: %s", path)
            shutil.rmtree(path)
    except Exception as e:
        logger.error("Failed to clean directory %s: %s", path, str(e))
        raise


def save_to_chroma(
    chunks: list[Document],
    chroma_db_dir: str,
    collection_name: str,
    model_name: str = SENTENCE_TRANSFORMERS_MODEL_NAME,
) -> bool:
    """Save document chunks to Chroma database."""
    if not chunks:
        logger.warning("No chunks to save to Chroma database")
        return False

    logger.info("Creating Chroma database for %s at %s...", collection_name, chroma_db_dir)

    try:
        clean_chroma_dir(chroma_db_dir)

        os.makedirs(os.path.dirname(chroma_db_dir), exist_ok=True)

        logger.info("Using model %s for embeddings.", model_name)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        db = Chroma.from_documents(
            chunks, embeddings, persist_directory=chroma_db_dir, collection_name=collection_name
        )

        collection_count = db._collection.count()
        if collection_count != len(chunks):
            logger.warning(
                "Expected %d chunks but database contains %d", len(chunks), collection_count
            )

        logger.info("Saved %d chunks to %s.", len(chunks), chroma_db_dir)
        logger.info("Chroma database for %s created at %s.", collection_name, chroma_db_dir)
        return True

    except Exception as e:
        logger.error("Failed to create Chroma database: %s", str(e))
        return False


def create_chroma_db(
    data_dir: str,
    chroma_db_dir: str,
    collection_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> bool:
    """Create Chroma database."""
    try:
        documents = load_documents(data_dir)
        if not documents:
            logger.error("No documents loaded, cannot create database")
            return False

        chunks = split_text(documents, chunk_size, chunk_overlap)
        if not chunks:
            logger.error("No chunks created, cannot create database")
            return False

        return save_to_chroma(chunks, chroma_db_dir, collection_name)

    except Exception as e:
        logger.error("Failed to create Chroma database: %s", str(e))
        return False


def get_database_configs() -> dict[str, tuple[str, str, str]]:
    """
    Get available database configurations.

    Returns:
        Dict mapping database type to (data_dir, chroma_db_dir, collection_name) tuple.
    """
    return {
        "books": (BOOKS_RAW_DATA_DIR, BOOKS_CHROMA_DB_DIR, BOOKS_COLLECTION_NAME),
        "azure": (AZURE_RAW_DATA_DIR, AZURE_CHROMA_DB_DIR, AZURE_COLLECTION_NAME),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create Chroma vector database from document collections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s books                    # Create books database with default settings
  %(prog)s azure                    # Create Azure docs database
  %(prog)s books --chunk-size 500   # Use custom chunk size
  %(prog)s books --chunk-overlap 50 # Use custom chunk overlap
        """,
    )

    # Available database types
    available_dbs = list(get_database_configs().keys())

    parser.add_argument(
        "database_type",
        choices=available_dbs,
        help=f"Type of database to create. Available options: {', '.join(available_dbs)}",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Size of text chunks for splitting (default: {DEFAULT_CHUNK_SIZE})",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between text chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=SENTENCE_TRANSFORMERS_MODEL_NAME,
        help=f"HuggingFace model name for embeddings (default: {SENTENCE_TRANSFORMERS_MODEL_NAME})",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging output"
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if args.chunk_size <= 0:
        raise ValueError(f"Chunk size must be positive, got: {args.chunk_size}")

    if args.chunk_overlap < 0:
        raise ValueError(f"Chunk overlap must be non-negative, got: {args.chunk_overlap}")

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError(
            f"Chunk overlap ({args.chunk_overlap}) must be less than chunk size ({args.chunk_size})"
        )


def main():
    """
    Main function to create Chroma database based on command line arguments.

    Supports multiple database types configured via constants.py:
    - books: Create database from book documents
    - azure: Create database from Azure documentation

    Command line options allow customization of chunk size, overlap, and model.
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)

        # Get database configuration
        configs = get_database_configs()
        data_dir, chroma_db_dir, collection_name = configs[args.database_type]

        # Log configuration
        logger.info("Creating Chroma database with configuration:")
        logger.info("  Database type: %s", args.database_type)
        logger.info("  Data directory: %s", data_dir)
        logger.info("  Chroma DB directory: %s", chroma_db_dir)
        logger.info("  Collection name: %s", collection_name)
        logger.info("  Chunk size: %d", args.chunk_size)
        logger.info("  Chunk overlap: %d", args.chunk_overlap)
        logger.info("  Model name: %s", args.model_name)

        # Create the database
        success = create_chroma_db(
            data_dir=data_dir,
            chroma_db_dir=chroma_db_dir,
            collection_name=collection_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        if not success:
            logger.error("Failed to create Chroma database for %s", args.database_type)
            sys.exit(1)

        logger.info("Successfully created Chroma database for %s", args.database_type)

    except ValueError as e:
        logger.error("Invalid arguments: %s", str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Database creation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
