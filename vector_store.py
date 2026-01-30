"""Vector store for document embeddings using ChromaDB."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import Config
from document_loader import Document
from embeddings import EmbeddingService


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(self, folder_path: str, embedding_service: EmbeddingService):
        self.folder_path = Path(folder_path).resolve()
        self.embedding_service = embedding_service
        self.index_path = self.folder_path / Config.INDEX_FOLDER

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.index_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list[Document], show_progress: bool = True):
        """Add documents to the vector store."""
        if not documents:
            return

        # Extract content and metadata
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        if show_progress:
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("Generating embeddings...", total=1)
                embeddings = self.embedding_service.embed_texts(contents)
                progress.update(task, completed=1)
        else:
            embeddings = self.embedding_service.embed_texts(contents)

        # Add to collection in batches
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=contents[i:end],
                metadatas=metadatas[i:end]
            )

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Search for similar documents."""
        top_k = top_k or Config.TOP_K_RESULTS

        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })

        return formatted_results

    def clear(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection("documents")
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()

    def exists(self) -> bool:
        """Check if the index exists and has documents."""
        return self.index_path.exists() and self.count() > 0
