"""Embedding services for text vectorization."""

from abc import ABC, abstractmethod
from typing import List

from config import Config


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their vectors."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbedding(EmbeddingService):
    """OpenAI embedding service."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_EMBEDDING_MODEL
        self._dimension = 1536  # text-embedding-3-small dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using OpenAI API."""
        if not texts:
            return []

        # Process in batches of 100
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=query
        )
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding(EmbeddingService):
    """Local embedding service using sentence-transformers."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(Config.LOCAL_EMBEDDING_MODEL)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using local model."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text."""
        embedding = self.model.encode(query)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_service(provider: str = None) -> EmbeddingService:
    """Factory function to get the appropriate embedding service."""
    provider = provider or Config.DEFAULT_EMBEDDING

    if provider == "openai":
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbedding()
    elif provider == "local":
        return LocalEmbedding()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
