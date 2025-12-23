"""Cohere embeddings service."""

from typing import Optional

import httpx

from ..config import settings


class EmbeddingsService:
    """Cohere embeddings integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.cohere_api_key
        self.model = model or settings.cohere_model
        self.base_url = "https://api.cohere.ai/v1"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def embed(
        self,
        texts: list[str],
        input_type: str = "search_document",
    ) -> list[list[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            input_type: One of "search_document", "search_query", "classification", "clustering"
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "texts": texts,
            "input_type": input_type,
        }

        response = await client.post("/embed", json=payload)
        response.raise_for_status()
        data = response.json()

        return data["embeddings"]

    async def embed_single(
        self,
        text: str,
        input_type: str = "search_document",
    ) -> list[float]:
        """Get embedding for a single text."""
        embeddings = await self.embed([text], input_type)
        return embeddings[0] if embeddings else []


# Global instance
embeddings_service = EmbeddingsService()

