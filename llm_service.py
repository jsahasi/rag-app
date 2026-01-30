"""LLM service abstraction for multiple providers."""

from abc import ABC, abstractmethod
from typing import Generator

from config import Config


class LLMService(ABC):
    """Abstract base class for LLM services."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the LLM provider."""
        pass


class AnthropicLLM(LLMService):
    """Anthropic Claude LLM service."""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model = Config.ANTHROPIC_MODEL

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a response using Claude."""
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """Generate a streaming response using Claude."""
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"


class OpenAILLM(LLMService):
    """OpenAI GPT LLM service."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a response using GPT."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """Generate a streaming response using GPT."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"


def get_llm_service(provider: str = None) -> LLMService:
    """Factory function to get the appropriate LLM service."""
    provider = provider or Config.DEFAULT_LLM

    if provider == "anthropic":
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic LLM")
        return AnthropicLLM()
    elif provider == "openai":
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI LLM")
        return OpenAILLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
