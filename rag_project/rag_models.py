"""
Pydantic models for RAG system data structures.
"""

import os
from typing import List, Optional

from pydantic import BaseModel, Field

DEFAULT_EMPTY_SOURCES: list = []
"""Default empty list for sources."""

DEFAULT_ZERO_RETRIEVED_DOCS: int = 0
"""Default number of retrieved documents when none found."""

DEFAULT_EMPTY_SCORES: list = []
"""Default empty list for similarity scores."""

DEFAULT_QUERY_DOCUMENTS_RETRIEVE: int = 3
"""Default number of documents to retrieve."""

DEFAULT_QUERY_MIN_DOCUMENTS_RETRIEVE: int = 1
"""Minimum number of documents to retrieve."""

DEFAULT_QUERY_MAX_DOCUMENTS_RETRIEVE: int = 20
"""Maximum number of documents to retrieve."""

DEFAULT_QUERY_MIN_SIMILARITY_SCORE: float = 0.5
"""Default minimum similarity score threshold."""

DEFAULT_QUERY_MIN_SCORE_THRESHOLD: float = 0.0
"""Minimum allowed similarity score threshold."""

DEFAULT_QUERY_MAX_SCORE_THRESHOLD: float = 1.0
"""Maximum allowed similarity score threshold."""

DEFAULT_QUERY_MAX_LENGTH: int = 500
"""Default maximum query length in characters."""

DEFAULT_QUERY_MIN_LENGTH: int = 1
"""Minimum allowed query length in characters."""

DEFAULT_QUERY_MAX_LENGTH_LIMIT: int = 2000
"""Maximum allowed query length limit."""

DEFAULT_QUERY_RETURN_SOURCES: bool = True
"""Default setting for returning document sources."""

DEFAULT_MODEL_NAME: str = "TheBloke/Llama-2-7B-Chat-GGUF"
"""Default model name for the LLM."""

DEFAULT_MODEL_TYPE: str = "llama"
"""Default model type."""

DEFAULT_MODEL_MAX_NEW_TOKENS: int = 256
"""Default maximum number of new tokens to generate."""

DEFAULT_MODEL_TEMPERATURE: float = 0.3
"""Default temperature for text generation."""

DEFAULT_MODEL_TOP_K: int = 40
"""Default top-k sampling parameter."""

DEFAULT_MODEL_TOP_P: float = 0.9
"""Default top-p (nucleus) sampling parameter."""

DEFAULT_MODEL_REPETITION_PENALTY: float = 1.05
"""Default repetition penalty."""

DEFAULT_MODEL_STOP_SEQUENCES: list = ["</s>", "[/INST]"]
"""Default stop sequences for text generation."""

DEFAULT_MODEL_CONTEXT_LENGTH: int = 2048
"""Default context length."""

DEFAULT_MODEL_THREADS: int = min(4, os.cpu_count() or 4)
"""Default number of threads for processing."""

DEFAULT_MODEL_SEED: int = 42
"""Default random seed."""

DEFAULT_MODEL_STREAM: bool = True
"""Default streaming setting."""

DEFAULT_PROMPT_TEMPLATE: str = """
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


class LLMConfig(BaseModel):
    """Configuration model for LLM parameters."""

    max_new_tokens: int = Field(
        default=DEFAULT_MODEL_MAX_NEW_TOKENS,
        ge=1,
        le=4096,
        description="Maximum number of new tokens to generate",
    )
    temperature: float = Field(
        default=DEFAULT_MODEL_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for text generation",
    )
    top_k: int = Field(
        default=DEFAULT_MODEL_TOP_K, ge=1, le=100, description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=DEFAULT_MODEL_TOP_P,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter",
    )
    repetition_penalty: float = Field(
        default=DEFAULT_MODEL_REPETITION_PENALTY, ge=1.0, le=2.0, description="Repetition penalty"
    )
    stop: List[str] = Field(
        default=DEFAULT_MODEL_STOP_SEQUENCES, description="Stop sequences for text generation"
    )
    context_length: int = Field(
        default=DEFAULT_MODEL_CONTEXT_LENGTH, ge=512, le=8192, description="Context length"
    )
    threads: int = Field(
        default=DEFAULT_MODEL_THREADS, ge=1, le=16, description="Number of threads for processing"
    )
    seed: int = Field(default=DEFAULT_MODEL_SEED, description="Random seed")
    stream: bool = Field(default=DEFAULT_MODEL_STREAM, description="Enable streaming")

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"


class RAGResponse(BaseModel):
    """Response model for RAG queries with validation."""

    answer: str = Field(..., description="Generated answer from the LLM")
    sources: List[Optional[str]] = Field(
        default_factory=list, description="List of source documents/files"
    )
    retrieved_docs: int = Field(
        ge=0, description="Number of documents retrieved from vector store"
    )
    similarity_scores: List[float] = Field(
        default_factory=list, description="Similarity scores for retrieved documents"
    )
    query: str = Field(..., description="Original user query")

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"


class QueryConfig(BaseModel):
    """Configuration model for RAG queries."""

    k: int = Field(
        default=DEFAULT_QUERY_DOCUMENTS_RETRIEVE,
        ge=DEFAULT_QUERY_MIN_DOCUMENTS_RETRIEVE,
        le=DEFAULT_QUERY_MAX_DOCUMENTS_RETRIEVE,
        description="Number of documents to retrieve",
    )
    min_similarity_score: float = Field(
        default=DEFAULT_QUERY_MIN_SIMILARITY_SCORE,
        ge=DEFAULT_QUERY_MIN_SCORE_THRESHOLD,
        le=DEFAULT_QUERY_MAX_SCORE_THRESHOLD,
        description="Minimum similarity score threshold",
    )
    max_length: int = Field(
        default=DEFAULT_QUERY_MAX_LENGTH,
        ge=DEFAULT_QUERY_MIN_LENGTH,
        le=DEFAULT_QUERY_MAX_LENGTH_LIMIT,
        description="Maximum query length in characters",
    )
    return_sources: bool = Field(
        default=DEFAULT_QUERY_RETURN_SOURCES, description="Whether to return document sources"
    )
    prompt_template: str = Field(
        default=DEFAULT_PROMPT_TEMPLATE, description="Custom prompt template"
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"


class EngineConfig(BaseModel):
    """Configuration model for RAG Engine initialization."""

    model: str = Field(default=DEFAULT_MODEL_NAME, description="Model name for the LLM")
    model_type: str = Field(default=DEFAULT_MODEL_TYPE, description="Model type (e.g., 'llama')")
    chroma_path: str = Field(description="Path to Chroma DB directory")
    collection_name: str = Field(description="Name of the Chroma collection")
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="LLM configuration parameters"
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
