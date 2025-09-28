from pathlib import Path
import time
from typing import Tuple

from langchain_chroma import Chroma
from langchain_community.llms import CTransformers

from rag_project.compare_embeddings import init_embeddings
from rag_project.constants import (
    LLAMA_MODEL_PATH,
)
from rag_project.logger import logger
from rag_project.rag_models import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_TYPE,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_QUERY_DOCUMENTS_RETRIEVE,
    DEFAULT_QUERY_MAX_LENGTH,
    DEFAULT_QUERY_MIN_SIMILARITY_SCORE,
    LLMConfig,
    RAGResponse,
)


def load_local_llama(
    model: str = DEFAULT_MODEL_NAME,
    model_type: str = DEFAULT_MODEL_TYPE,
    config: LLMConfig = LLMConfig(),
) -> CTransformers:
    """
    Initialize and return a local LLM using CTransformers.

    Args:
        model: Model name for the LLM
        model_type: Model type (e.g., 'llama')
        config: Optional LLMConfig instance to override defaults

    Returns:
        CTransformers: Configured LLM instance

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model initialization fails
    """
    start_time = time.time()

    logger.info(
        "load_local_llama called with parameters: model='%s', model_type='%s', config=%s",
        model,
        model_type,
        config.model_dump() if config else None,
    )

    model_path = Path(LLAMA_MODEL_PATH)
    logger.info("Checking model file existence at path: %s", model_path)

    if not model_path.exists():
        error_msg = (
            f"Model file not found at {model_path}. "
            "Please download it first using: \n"
            "make get_llama_model"
        )
        logger.error("Model file validation failed: %s", error_msg)
        execution_time = time.time() - start_time
        logger.error("load_local_llama failed after %.3f seconds", execution_time)
        raise FileNotFoundError(error_msg)

    try:
        logger.info("Initializing CTransformers with model file: %s", model_path)
        llm = CTransformers(
            model=model,
            model_file=str(model_path),
            model_type=model_type,
            config=config.model_dump(),
        )

        execution_time = time.time() - start_time
        logger.info(
            "load_local_llama completed successfully in %.3f seconds. LLM type: %s, Model: %s",
            execution_time,
            type(llm).__name__,
            model,
        )

        return llm

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "Failed to initialize LLM after %.3f seconds. Error: %s. "
            "Parameters: model='%s', model_type='%s', model_path='%s'",
            execution_time,
            str(e),
            model,
            model_type,
            model_path,
        )
        raise


def init_chroma(chroma_str_path: str, chroma_collection_name: str) -> Chroma:
    """
    Initialize and return Chroma DB.

    Args:
        chroma_str_path: Path to the Chroma DB directory
        chroma_collection_name: Name of the Chroma collection

    Returns:
        Chroma: Configured Chroma DB instance

    Raises:
        Exception: If Chroma initialization fails
    """
    start_time = time.time()

    logger.info(
        "init_chroma called with parameters: chroma_str_path='%s', collection_name='%s'",
        chroma_str_path,
        chroma_collection_name,
    )

    try:
        logger.info("Initializing embeddings function")
        embedding_function = init_embeddings()
        logger.info("Embeddings function initialized successfully")

        chroma_path = Path(chroma_str_path)
        logger.info("Checking Chroma directory path: %s", chroma_path)

        if not chroma_path.exists():
            logger.warning("Chroma directory doesn't exist: %s. Creating it.", chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            logger.info("Created Chroma directory: %s", chroma_path)

        logger.info(
            "Initializing Chroma DB at %s with collection '%s'",
            chroma_path,
            chroma_collection_name,
        )

        chroma_db = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embedding_function,
            collection_name=chroma_collection_name,
        )

        execution_time = time.time() - start_time
        logger.info(
            "init_chroma completed successfully in %.3f seconds. "
            "Collection: %s, Path: %s, DB type: %s",
            execution_time,
            chroma_collection_name,
            chroma_path,
            type(chroma_db).__name__,
        )

        return chroma_db

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "Failed to initialize Chroma DB after %.3f seconds. Error: %s. "
            "Parameters: path='%s', collection='%s'",
            execution_time,
            str(e),
            chroma_str_path,
            chroma_collection_name,
        )
        raise


def format_prompt(
    context: str, question: str, prompt_template: str = DEFAULT_PROMPT_TEMPLATE
) -> str:
    """
    Format the prompt with context and question.

    Args:
        context: Retrieved context documents
        question: User question
        template: Optional custom prompt template

    Returns:
        str: Formatted prompt
    """
    start_time = time.time()

    context_preview = context[:100] + "..." if len(context) > 100 else context
    question_preview = question[:100] + "..." if len(question) > 100 else question

    logger.info(
        "format_prompt called with parameters: context_length=%d, question_length=%d, "
        "template_length=%d, context_preview='%s', question_preview='%s'",
        len(context),
        len(question),
        len(prompt_template),
        context_preview,
        question_preview,
    )

    try:
        if not context.strip():
            logger.warning("Empty context provided for prompt formatting")

        if not question.strip():
            error_msg = "Question cannot be empty"
            execution_time = time.time() - start_time
            logger.error(
                "format_prompt validation failed after %.3f seconds: %s", execution_time, error_msg
            )
            raise ValueError(error_msg)

        formatted_prompt = prompt_template.format(context=context, question=question)

        execution_time = time.time() - start_time
        logger.info(
            "format_prompt completed successfully in %.3f seconds. Output length: %d characters",
            execution_time,
            len(formatted_prompt),
        )

        return formatted_prompt

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "format_prompt failed after %.3f seconds. Error: %s. "
            "Parameters: context_length=%d, question_length=%d",
            execution_time,
            str(e),
            len(context),
            len(question),
        )
        raise


def validate_query_inputs(question: str, max_length: int = DEFAULT_QUERY_MAX_LENGTH) -> None:
    """
    Validate query inputs.

    Args:
        question: User question to validate
        max_length: Maximum allowed question length

    Raises:
        ValueError: If validation fails
    """
    start_time = time.time()

    question_preview = question[:100] + "..." if len(question) > 100 else question
    logger.info(
        "validate_query_inputs called with parameters: question_length=%d, max_length=%d, "
        "question_preview='%s'",
        len(question) if question else 0,
        max_length,
        question_preview,
    )

    try:
        if not question or not question.strip():
            error_msg = "Question cannot be empty"
            execution_time = time.time() - start_time
            logger.error(
                "validate_query_inputs failed after %.3f seconds: %s", execution_time, error_msg
            )
            raise ValueError(error_msg)

        if len(question) > max_length:
            error_msg = f"Question too long. Maximum {max_length} characters allowed"
            execution_time = time.time() - start_time
            logger.error(
                "validate_query_inputs failed after %.3f seconds: %s. Question length: %d",
                execution_time,
                error_msg,
                len(question),
            )
            raise ValueError(error_msg)

        execution_time = time.time() - start_time
        logger.info(
            "validate_query_inputs completed successfully in %.3f seconds. "
            "Question validated: length=%d",
            execution_time,
            len(question),
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "validate_query_inputs encountered unexpected error after %.3f seconds: %s. "
            "Parameters: question_length=%d, max_length=%d",
            execution_time,
            str(e),
            len(question) if question else 0,
            max_length,
        )
        raise


class RAGQueryEngine:
    """
    RAG Query Engine for handling document retrieval and generation.

    This class encapsulates the complete RAG pipeline including document retrieval
    from Chroma DB and answer generation using a local LLM.
    """

    def __init__(self, llm: CTransformers, chroma: Chroma, engine_config: dict):
        """
        Initialize RAG Query Engine.

        Args:
            llm (CTransformers): Pre-configured CTransformers LLM instance for text generation.
            chroma (Chroma): Pre-configured Chroma vector database instance for document retrieval.
            engine_config (dict): Configuration dictionary containing engine metadata with keys:
                - model (str): Model name for the LLM
                - model_type (str): Model type (e.g., 'llama')
                - chroma_path (str): Path to Chroma DB directory
                - collection_name (str): Name of the Chroma collection
                - llm_config (dict): LLM configuration parameters

        Raises:
            Exception: If initialization fails during attribute assignment or logging setup.

        Note:
            This constructor expects pre-initialized LLM and Chroma instances. Use the
            create_rag_engine() factory function for automatic initialization of all components.
        """
        start_time = time.time()

        logger.info(
            "RAGQueryEngine.__init__ called with parameters: llm_type='%s', chroma_type='%s', "
            "engine_config_keys=%s",
            type(llm).__name__,
            type(chroma).__name__,
            list(engine_config.keys()),
        )

        try:
            self.llm = llm
            self.chroma = chroma
            self.model = engine_config.get("model")
            self.model_type = engine_config.get("model_type")
            self.chroma_path = engine_config.get("chroma_path")
            self.collection_name = engine_config.get("collection_name")
            self.llm_config = engine_config.get("llm_config")

            execution_time = time.time() - start_time
            logger.info(
                "RAG Query Engine initialized successfully in %.3f seconds. "
                "Model: %s, Collection: %s, Path: %s",
                execution_time,
                self.model,
                self.collection_name,
                self.chroma_path,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Failed to initialize RAG Query Engine after %.3f seconds: %s. Engine config: %s",
                execution_time,
                str(e),
                engine_config,
            )
            raise

    def retrieve_documents(
        self,
        question: str,
        documents_retrieve: int = DEFAULT_QUERY_DOCUMENTS_RETRIEVE,
        min_similarity_score: float = DEFAULT_QUERY_MIN_SIMILARITY_SCORE,
    ) -> list:
        """
        Retrieve relevant documents from Chroma DB with relevance filtering.

        Args:
            question: User question for similarity search
            documents_retrieve: Number of documents to retrieve
            min_similarity_score: Minimum relevance score threshold

        Returns:
            List[Document]: Retrieved documents above the relevance threshold
        """
        start_time = time.time()

        question_preview = question[:100] + "..." if len(question) > 100 else question
        logger.info(
            "retrieve_documents called with parameters: question_length=%d, "
            "documents_retrieve=%d, min_similarity_score=%.3f, question_preview='%s'",
            len(question),
            documents_retrieve,
            min_similarity_score,
            question_preview,
        )

        if documents_retrieve <= 0:
            error_msg = "documents_retrieve must be positive"
            execution_time = time.time() - start_time
            logger.error(
                "retrieve_documents validation failed after %.3f seconds: %s. "
                "documents_retrieve=%d",
                execution_time,
                error_msg,
                documents_retrieve,
            )
            raise ValueError(error_msg)

        try:
            logger.info("Performing similarity search with Chroma DB")
            results = self.chroma.similarity_search_with_relevance_scores(
                question, k=documents_retrieve
            )

            logger.info(
                "Filtering documents by similarity score threshold: %.3f", min_similarity_score
            )
            filtered_docs = [doc for doc, score in results if score >= min_similarity_score]

            execution_time = time.time() - start_time
            logger.info(
                "retrieve_documents completed successfully in %.3f seconds. "
                "Retrieved %d documents (filtered from %d total) for query",
                execution_time,
                len(filtered_docs),
                len(results),
            )

            scores = [score for doc, score in results if score >= min_similarity_score]
            logger.info("Document similarity scores: %s", scores)

            return filtered_docs

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Failed to retrieve documents after %.3f seconds: %s. "
                "Parameters: question_length=%d, documents_retrieve=%d, min_similarity_score=%.3f",
                execution_time,
                str(e),
                len(question),
                documents_retrieve,
                min_similarity_score,
            )
            raise

    def retrieve_documents_with_scores(
        self,
        question: str,
        documents_retrieve: int = DEFAULT_QUERY_DOCUMENTS_RETRIEVE,
        min_similarity_score: float = DEFAULT_QUERY_MIN_SIMILARITY_SCORE,
    ) -> Tuple[list, list]:
        """
        Retrieve relevant documents from Chroma DB with relevance filtering and scores.

        Args:
            question: User question for similarity search
            documents_retrieve: Number of documents to retrieve
            min_similarity_score: Minimum relevance score threshold

        Returns:
            Tuple[List[Document], List[float]]: Retrieved documents and their scores
        """
        start_time = time.time()

        question_preview = question[:100] + "..." if len(question) > 100 else question
        logger.info(
            "retrieve_documents_with_scores called with parameters: question_length=%d, "
            "documents_retrieve=%d, min_similarity_score=%.3f, question_preview='%s'",
            len(question),
            documents_retrieve,
            min_similarity_score,
            question_preview,
        )

        if documents_retrieve <= 0:
            error_msg = "documents_retrieve must be positive"
            execution_time = time.time() - start_time
            logger.error(
                "retrieve_documents_with_scores validation failed after %.3f seconds: %s. "
                "documents_retrieve=%d",
                execution_time,
                error_msg,
                documents_retrieve,
            )
            raise ValueError(error_msg)

        try:
            logger.info("Performing similarity search with Chroma DB (with scores)")
            results = self.chroma.similarity_search_with_relevance_scores(
                question, k=documents_retrieve
            )

            logger.info(
                "Filtering documents and scores by similarity threshold: %.3f",
                min_similarity_score,
            )
            filtered_results = [
                (doc, score) for doc, score in results if score >= min_similarity_score
            ]
            filtered_docs = [doc for doc, score in filtered_results]
            filtered_scores = [score for doc, score in filtered_results]

            execution_time = time.time() - start_time
            logger.info(
                "retrieve_documents_with_scores completed successfully in %.3f seconds. "
                "Retrieved %d documents (filtered from %d total) with scores: %s",
                execution_time,
                len(filtered_docs),
                len(results),
                filtered_scores,
            )

            return filtered_docs, filtered_scores

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Failed to retrieve documents with scores after %.3f seconds: %s. "
                "Parameters: question_length=%d, documents_retrieve=%d, min_similarity_score=%.3f",
                execution_time,
                str(e),
                len(question),
                documents_retrieve,
                min_similarity_score,
            )
            raise

    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using the LLM.

        Args:
            prompt: Formatted prompt with context and question

        Returns:
            str: Generated answer

        Raises:
            Exception: If generation fails
        """
        start_time = time.time()

        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        logger.info(
            "generate_answer called with parameters: prompt_length=%d, prompt_preview='%s'",
            len(prompt),
            prompt_preview,
        )

        try:
            logger.info("Invoking LLM for answer generation")
            answer = self.llm.invoke(prompt)

            execution_time = time.time() - start_time
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            logger.info(
                "generate_answer completed successfully in %.3f seconds. "
                "Answer length: %d, answer_preview='%s'",
                execution_time,
                len(answer),
                answer_preview,
            )

            return answer

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Failed to generate answer after %.3f seconds: %s. Prompt length: %d",
                execution_time,
                str(e),
                len(prompt),
            )
            raise

    def query(
        self,
        question: str,
        documents_retrieve: int = DEFAULT_QUERY_DOCUMENTS_RETRIEVE,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_length: int = DEFAULT_QUERY_MAX_LENGTH,
        min_similarity_score: float = DEFAULT_QUERY_MIN_SIMILARITY_SCORE,
    ) -> str:
        """
        Query the RAG system with a question.

        Args:
            question: User question
            documents_retrieve: Number of documents to retrieve (default: 3)
            prompt_template: Optional custom prompt template
            max_length: Maximum question length (default: 500)

        Returns:
            str: Generated answer

        Raises:
            ValueError: If question is invalid
            Exception: If query processing fails
        """
        start_time = time.time()

        question_preview = question[:100] + "..." if len(question) > 100 else question
        logger.info(
            "query called with parameters: question_length=%d, documents_retrieve=%d, "
            "max_length=%d, min_similarity_score=%.3f, question_preview='%s'",
            len(question),
            documents_retrieve,
            max_length,
            min_similarity_score,
            question_preview,
        )

        try:
            logger.info("Validating query inputs")
            validate_query_inputs(question, max_length)

            if documents_retrieve <= 0:
                error_msg = "Number of documents (documents_retrieve) must be positive"
                execution_time = time.time() - start_time
                logger.error(
                    "query validation failed after %.3f seconds: %s. documents_retrieve=%d",
                    execution_time,
                    error_msg,
                    documents_retrieve,
                )
                raise ValueError(error_msg)

            logger.info("Processing query: %s...", question[:50])

            logger.info("Retrieving documents")
            docs = self.retrieve_documents(question, documents_retrieve, min_similarity_score)

            if not docs:
                execution_time = time.time() - start_time
                fallback_answer = (
                    "I couldn't find any relevant information to answer your question."
                )
                logger.warning(
                    "No relevant documents retrieved for query after %.3f seconds. "
                    "Returning fallback answer: '%s'",
                    execution_time,
                    fallback_answer,
                )
                return fallback_answer

            logger.info("Building context from %d retrieved documents", len(docs))
            context = "\n\n".join([doc.page_content for doc in docs])

            template = prompt_template or DEFAULT_PROMPT_TEMPLATE
            logger.info("Formatting prompt with template")
            prompt = format_prompt(context, question, template)

            logger.info("Generating answer with LLM")
            answer = self.generate_answer(prompt)

            execution_time = time.time() - start_time
            final_answer = answer.strip()
            logger.info(
                "query completed successfully in %.3f seconds. Answer length: %d, "
                "documents_used: %d, question: %s...",
                execution_time,
                len(final_answer),
                len(docs),
                question[:50],
            )

            return final_answer

        except ValueError as e:
            execution_time = time.time() - start_time
            logger.error(
                "query validation error after %.3f seconds: %s. Question: %s...",
                execution_time,
                str(e),
                question[:50],
            )
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Error processing query after %.3f seconds: %s. Question: %s..., "
                "Parameters: documents_retrieve=%d, max_length=%d, min_similarity_score=%.3f",
                execution_time,
                str(e),
                question[:50],
                documents_retrieve,
                max_length,
                min_similarity_score,
            )
            raise RuntimeError(f"Failed to process query: {str(e)}") from e

    def query_with_metadata(
        self,
        question: str,
        documents_retrieve: int = DEFAULT_QUERY_DOCUMENTS_RETRIEVE,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_length: int = DEFAULT_QUERY_MAX_LENGTH,
        min_similarity_score: float = DEFAULT_QUERY_MIN_SIMILARITY_SCORE,
        return_sources: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG system with a question and return complete metadata.
        This implements the functionality from the notebook results section.

        Args:
            question: User question
            documents_retrieve: Number of documents to retrieve
            prompt_template: Optional custom prompt template
            max_length: Maximum question length
            min_similarity_score: Minimum similarity score threshold
            return_sources: Whether to include source information

        Returns:
            RAGResponse: Complete response with answer, sources, and metadata

        Raises:
            ValueError: If question is invalid
            Exception: If query processing fails
        """
        start_time = time.time()

        question_preview = question[:100] + "..." if len(question) > 100 else question
        logger.info(
            "query_with_metadata called with parameters: question_length=%d, "
            "documents_retrieve=%d, max_length=%d, min_similarity_score=%.3f, "
            "return_sources=%s, question_preview='%s'",
            len(question),
            documents_retrieve,
            max_length,
            min_similarity_score,
            return_sources,
            question_preview,
        )

        try:
            logger.info("Validating query inputs for metadata query")
            validate_query_inputs(question, max_length)

            if documents_retrieve <= 0:
                error_msg = "Number of documents (documents_retrieve) must be positive"
                execution_time = time.time() - start_time
                logger.error(
                    "query_with_metadata validation failed after %.3f seconds: %s. "
                    "documents_retrieve=%d",
                    execution_time,
                    error_msg,
                    documents_retrieve,
                )
                raise ValueError(error_msg)

            logger.info("Processing query with metadata: %s...", question[:50])

            logger.info("Retrieving documents with scores")
            docs, scores = self.retrieve_documents_with_scores(
                question, documents_retrieve, min_similarity_score
            )

            if not docs:
                execution_time = time.time() - start_time
                fallback_response = RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    retrieved_docs=0,
                    similarity_scores=[],
                    query=question,
                )
                logger.warning(
                    "No relevant documents retrieved for metadata query after %.3f seconds. "
                    "Returning fallback response",
                    execution_time,
                )
                return fallback_response

            logger.info("Building context from %d retrieved documents", len(docs))
            context = "\n\n".join([doc.page_content for doc in docs])

            template = prompt_template or DEFAULT_PROMPT_TEMPLATE
            logger.info("Formatting prompt with template for metadata query")
            prompt = format_prompt(context, question, template)

            logger.info("Generating answer with LLM for metadata query")
            answer = self.generate_answer(prompt)

            sources = []
            if return_sources:
                logger.info("Extracting sources from document metadata")
                sources = [doc.metadata.get("source", None) for doc in docs]
                logger.info("Extracted %d sources: %s", len(sources), sources)

            response = RAGResponse(
                answer=answer.strip(),
                sources=sources,
                retrieved_docs=len(docs),
                similarity_scores=scores,
                query=question,
            )

            execution_time = time.time() - start_time
            logger.info(
                "query_with_metadata completed successfully in %.3f seconds. "
                "Answer length: %d, documents_used: %d, sources: %d, "
                "similarity_scores: %s, question: %s...",
                execution_time,
                len(response.answer),
                response.retrieved_docs,
                len(response.sources),
                response.similarity_scores,
                question[:50],
            )

            return response

        except ValueError as e:
            execution_time = time.time() - start_time
            logger.error(
                "query_with_metadata validation error after %.3f seconds: %s. Question: %s...",
                execution_time,
                str(e),
                question[:50],
            )
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Error processing query with metadata after %.3f seconds: %s. "
                "Question: %s..., Parameters: documents_retrieve=%d, max_length=%d, "
                "min_similarity_score=%.3f, return_sources=%s",
                execution_time,
                str(e),
                question[:50],
                documents_retrieve,
                max_length,
                min_similarity_score,
                return_sources,
            )
            raise RuntimeError(f"Failed to process query: {str(e)}") from e

    def get_config_info(self) -> dict:
        """
        Get current configuration information.

        Returns:
            dict: Configuration details
        """
        start_time = time.time()

        logger.info("get_config_info called")

        try:
            config = {
                "model": self.model,
                "model_type": self.model_type,
                "chroma_path": self.chroma_path,
                "collection_name": self.collection_name,
                "llm_config": self.llm_config,
            }

            execution_time = time.time() - start_time
            logger.info(
                "get_config_info completed successfully in %.3f seconds. Config keys: %s",
                execution_time,
                list(config.keys()),
            )

            return config

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Error getting config info after %.3f seconds: %s", execution_time, str(e)
            )
            raise

    def __repr__(self) -> str:
        """String representation of the RAG Query Engine."""
        return f"RAGQueryEngine(model='{self.model}', collection='{self.collection_name}')"


def create_rag_engine(
    chroma_path: str,
    collection_name: str,
    model: str = DEFAULT_MODEL_NAME,
    model_type: str = DEFAULT_MODEL_TYPE,
    llm_config: LLMConfig = LLMConfig(),
) -> RAGQueryEngine:
    """
    Create and configure a complete RAG (Retrieval-Augmented Generation) query engine.

    This factory function initializes all components needed for a RAG system:
    - Local LLM using CTransformers with specified model
    - Chroma vector database for document retrieval
    - Complete RAGQueryEngine instance ready for querying

    Args:
        chroma_path (str): Path to the Chroma database directory. Will be created if it doesn't exist.
        collection_name (str): Name of the Chroma collection to use for document storage/retrieval.
        model (str, optional): Model name for the LLM. Defaults to DEFAULT_MODEL_NAME.
        model_type (str, optional): Model type (e.g., 'llama'). Defaults to DEFAULT_MODEL_TYPE.
        llm_config (LLMConfig, optional): Configuration object for LLM parameters
            (temperature, max_tokens, etc.). Defaults to LLMConfig().

    Returns:
        RAGQueryEngine: Fully configured RAG query engine ready for document retrieval
            and answer generation.

    Raises:
        FileNotFoundError: If the specified model file doesn't exist at LLAMA_MODEL_PATH.
        Exception: If LLM initialization fails or Chroma database setup encounters errors.

    Example:
        >>> engine = create_rag_engine(
        ...     chroma_path="./data/chroma-db/books",
        ...     collection_name="alice_in_wonderland",
        ...     model="llama-2-7b-chat.Q4_K_M.gguf"
        ... )
        >>> response = engine.query("What is the story about?")
    """
    start_time = time.time()

    logger.info(
        "create_rag_engine called with parameters: chroma_path='%s', "
        "collection_name='%s', model='%s', model_type='%s', llm_config=%s",
        chroma_path,
        collection_name,
        model,
        model_type,
        llm_config.model_dump(),
    )

    try:
        logger.info("Initializing LLM component")
        llm = load_local_llama(model, model_type, llm_config)

        logger.info("Initializing Chroma DB component")
        chroma = init_chroma(chroma_path, collection_name)

        logger.info("Building engine configuration")
        engine_config = {
            "model": model,
            "model_type": model_type,
            "chroma_path": chroma_path,
            "collection_name": collection_name,
            "llm_config": llm_config.model_dump(),
        }

        logger.info("Creating RAGQueryEngine instance")
        engine = RAGQueryEngine(llm=llm, chroma=chroma, engine_config=engine_config)

        execution_time = time.time() - start_time
        logger.info(
            "create_rag_engine completed successfully in %.3f seconds. "
            "Engine: %s, Model: %s, Collection: %s",
            execution_time,
            type(engine).__name__,
            model,
            collection_name,
        )

        return engine

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "Failed to create RAG engine after %.3f seconds: %s. "
            "Parameters: chroma_path='%s', collection_name='%s', model='%s', model_type='%s'",
            execution_time,
            str(e),
            chroma_path,
            collection_name,
            model,
            model_type,
        )
        raise
