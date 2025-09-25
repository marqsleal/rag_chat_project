import os
import sys

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag_project.constants import (
    BOOKS_CHROMA_DB_DIR,
    BOOKS_COLLECTION_NAME,
    BOOKS_RAW_DATA_DIR,
    SENTENCE_TRANSFORMERS_MODEL_NAME,
)
from rag_project.logger import logger

load_dotenv()


def load_documents(directory: str) -> list[Document]:
    logger.info("Loading documents from %s...", directory)
    loader = DirectoryLoader(directory, glob="**/*.md")
    documents = loader.load()
    logger.info("Loaded %d documents from %s.", len(documents), directory)
    return documents


def split_text(documents: list[Document] = None) -> list[Document]:
    logger.info("Splitting text from %d documents...", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))

    document = chunks[10]
    logger.info("Document content example: %s", document.page_content)
    logger.info("Document metadata example: %s", document.metadata)

    return chunks


def clean_chroma_dir(path):
    for root, repo_dirs, repo_files in os.walk(path, topdown=False):
        for repo_file in repo_files:
            file_path = os.path.join(root, repo_file)
            os.remove(file_path)
        for repo_dir in repo_dirs:
            dir_path = os.path.join(root, repo_dir)
            os.rmdir(dir_path)
    os.rmdir(path)


def save_to_chroma(
    chunks: list[Document],
    chroma_db_dir: str,
    collection_name: str,
    model_name=SENTENCE_TRANSFORMERS_MODEL_NAME
) -> bool:
    logger.info("Creating Chroma database for %s at %s...", collection_name, chroma_db_dir)

    if os.path.exists(chroma_db_dir):
        logger.info("Cleaning existing Chroma directory at %s...", chroma_db_dir)
        clean_chroma_dir(chroma_db_dir)

    logger.info("Using model %s for embeddings.", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=chroma_db_dir,
        collection_name=collection_name
    )
    db.persist()
    logger.info("Saved %d chunks to %s.", len(chunks), chroma_db_dir)
    logger.info("Chroma database for %s created at %s.", collection_name, chroma_db_dir)
    return True


def create_chroma_db(data_dir: str, chroma_db_dir: str, collection_name: str):
    documents = load_documents(data_dir)
    chunks = split_text(documents)
    if not save_to_chroma(chunks, chroma_db_dir, collection_name):
        sys.exit(1)


def main():
    create_chroma_db(BOOKS_RAW_DATA_DIR, BOOKS_CHROMA_DB_DIR, BOOKS_COLLECTION_NAME)


if __name__ == "__main__":
    main()