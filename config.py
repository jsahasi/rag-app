"""Configuration management for RAG application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Defaults
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "anthropic")
    DEFAULT_EMBEDDING: str = os.getenv("DEFAULT_EMBEDDING", "local")

    # Embedding models
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # LLM models
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    OPENAI_MODEL: str = "gpt-4o"

    # Chunking settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # RAG settings
    TOP_K_RESULTS: int = 5

    # File extensions
    TEXT_EXTENSIONS: set = {".txt", ".md", ".markdown"}
    CODE_EXTENSIONS: set = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
        ".html", ".css", ".scss", ".sass", ".xml", ".csv", ".sql",
        ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
        ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb",
        ".php", ".swift", ".kt", ".scala", ".r", ".m", ".mm"
    }
    PDF_EXTENSIONS: set = {".pdf"}
    DOCX_EXTENSIONS: set = {".docx"}

    # Index folder name
    INDEX_FOLDER: str = ".rag_index"
    INSTRUCTIONS_FILE: str = "instructions.txt"

    @classmethod
    def get_supported_extensions(cls) -> set:
        """Get all supported file extensions."""
        return (
            cls.TEXT_EXTENSIONS |
            cls.CODE_EXTENSIONS |
            cls.PDF_EXTENSIONS |
            cls.DOCX_EXTENSIONS
        )

    @classmethod
    def validate_llm_config(cls, llm_provider: str) -> bool:
        """Validate that required API key is set for the LLM provider."""
        if llm_provider == "anthropic":
            return bool(cls.ANTHROPIC_API_KEY)
        elif llm_provider == "openai":
            return bool(cls.OPENAI_API_KEY)
        return False

    @classmethod
    def validate_embedding_config(cls, embedding_provider: str) -> bool:
        """Validate that required API key is set for the embedding provider."""
        if embedding_provider == "openai":
            return bool(cls.OPENAI_API_KEY)
        elif embedding_provider == "local":
            return True  # Local embeddings don't need API key
        return False
