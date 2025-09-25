import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.llms import CTransformers

from rag_project.compare_embeddings import init_embeddings
from rag_project.constants import (
    BOOKS_CHROMA_DB_DIR,
    BOOKS_COLLECTION_NAME,
    LLAMA_MODEL_PATH,
)


PROMPT_TEMPLATE: str = """
[INST] <<SYS>>
You are a clear and concise assistant specialized in summarization and explanation.
Use only the provided context to answer the question.
If the context does not provide enough information, reply with:
"The context does not contain enough information to answer."
Avoid speculation.
<</SYS>>

Context:
{context}

Question:
{question}
[/INST]
"""
"""Prompt template for RAG with context and question."""


def load_local_llama() -> CTransformers:
    """Initialize and return a local LLM using CTransformers.""" 
    model_path = Path(LLAMA_MODEL_PATH)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please download it first using: \n"
            "make get_llama_model"
        )
    
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_file=str(model_path),
        model_type="llama",
        config={
            "max_new_tokens": 256,
            "temperature": 0.3,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "stop": ["</s>", "[/INST]"],
            "context_length": 2048,
            "threads": min(4, os.cpu_count()),
            "seed": 42,
            "stream": True
        }
    )


def init_chroma() -> Chroma:
    """Initialize and return Chroma DB."""
    embedding_function = init_embeddings()
    chroma_path = str(Path(BOOKS_CHROMA_DB_DIR))
    return Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function,
        collection_name=BOOKS_COLLECTION_NAME
    )