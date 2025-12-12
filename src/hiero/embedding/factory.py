from __future__ import annotations

from hiero.config import Settings

from .base import EmbedderProtocol
from .openai import OpenAIEmbedder


class EmbedderFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create(self) -> EmbedderProtocol:
        provider = self.settings.default_embedding_provider.lower()
        if provider != "openai":
            raise ValueError(f"Unsupported provider for MVP: {provider}")
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbedder(
            api_key=self.settings.openai_api_key.get_secret_value(),
            model_name=self.settings.default_embedding_model,
        )

