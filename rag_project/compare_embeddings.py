from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from rag_project.constants import (
    SENTENCE_TRANSFORMERS_MODEL_NAME,
)


def init_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return HuggingFace embeddings model."""
    model_name = SENTENCE_TRANSFORMERS_MODEL_NAME
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"batch_size": 8, "normalize_embeddings": True, "convert_to_numpy": True}

    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def get_embedding(text: str, model: HuggingFaceEmbeddings) -> list:
    """Get embeddings for a text using HuggingFace model."""
    embeddings = model.embed_documents([text])
    return embeddings[0]


def compare_embeddings(text1: str, text2: str, model: HuggingFaceEmbeddings) -> float:
    """Compare two texts using cosine similarity."""
    embeddings1 = get_embedding(text1, model)
    embeddings2 = get_embedding(text2, model)
    similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
    return float(similarity)
