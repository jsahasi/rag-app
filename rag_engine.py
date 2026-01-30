"""RAG engine that orchestrates the retrieval-augmented generation pipeline."""

from typing import Generator

from config import Config
from document_loader import DocumentLoader, get_instructions
from embeddings import get_embedding_service, EmbeddingService
from vector_store import VectorStore
from llm_service import get_llm_service, LLMService


class RAGEngine:
    """Orchestrates the RAG pipeline."""

    def __init__(
        self,
        folder_path: str,
        llm_provider: str = None,
        embedding_provider: str = None
    ):
        self.folder_path = folder_path
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

        # Initialize services
        self.embedding_service: EmbeddingService = get_embedding_service(embedding_provider)
        self.llm_service: LLMService = get_llm_service(llm_provider)
        self.vector_store: VectorStore = VectorStore(folder_path, self.embedding_service)

        # Load instructions if available
        self.instructions = get_instructions(folder_path)

    def index_documents(self, rebuild: bool = False) -> int:
        """Index all documents in the folder."""
        if rebuild:
            self.vector_store.clear()

        # Load documents
        loader = DocumentLoader(self.folder_path)
        documents = loader.load_all()

        if not documents:
            return 0

        # Clear existing and add new
        if not rebuild and self.vector_store.count() > 0:
            self.vector_store.clear()

        # Add documents to vector store
        self.vector_store.add_documents(documents)

        return len(documents)

    def query(self, question: str, stream: bool = False) -> str | Generator[str, None, None]:
        """Query the RAG system with a question."""
        # Retrieve relevant documents
        results = self.vector_store.search(question, top_k=Config.TOP_K_RESULTS)

        # Build context from retrieved documents
        context = self._build_context(results)

        # Build the prompt
        prompt = self._build_prompt(question, context)

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Generate response
        if stream:
            return self.llm_service.generate_stream(prompt, system_prompt)
        else:
            return self.llm_service.generate(prompt, system_prompt)

    def _build_context(self, results: list[dict]) -> str:
        """Build context string from search results."""
        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            content = result["content"]
            context_parts.append(f"[Document {i} - {source}]\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the user prompt with context and question."""
        return f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so clearly.

## Context

{context}

## Question

{question}

## Answer"""

    def _build_system_prompt(self) -> str:
        """Build the system prompt, incorporating instructions if available."""
        base_prompt = "You are a helpful assistant that answers questions based on the provided context documents."

        if self.instructions:
            return f"{self.instructions}\n\nAdditional guidelines:\n- Always base your answers on the provided context\n- If information is not in the context, clearly state that\n- Cite the source documents when relevant"
        else:
            return f"{base_prompt}\n\n- Always base your answers on the provided context\n- If information is not in the context, clearly state that\n- Cite the source documents when relevant"

    def is_indexed(self) -> bool:
        """Check if the folder has been indexed."""
        return self.vector_store.exists()

    def document_count(self) -> int:
        """Get the number of indexed document chunks."""
        return self.vector_store.count()
