"""OpenRouter LLM service with streaming support."""

import json
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field

import httpx

from ..config import settings


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None


class LLMService:
    """OpenRouter LLM integration with streaming support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = base_url or settings.openrouter_base_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://ambient-desktop.local",
                    "X-Title": "Ambient Desktop Agent",
                },
                timeout=120.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Get a completion from the LLM."""
        client = await self._get_client()
        
        # Build messages list
        msg_list = []
        if system_prompt:
            msg_list.append({"role": "system", "content": system_prompt})
        for msg in messages:
            msg_list.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": model or self.model,
            "messages": msg_list,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from the LLM."""
        client = await self._get_client()
        
        # Build messages list
        msg_list = []
        if system_prompt:
            msg_list.append({"role": "system", "content": system_prompt})
        for msg in messages:
            msg_list.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": model or self.model,
            "messages": msg_list,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def parse_task(
        self,
        user_instruction: str,
        context: Optional[str] = None,
    ) -> dict:
        """Parse a user instruction into structured tasks."""
        system_prompt = """You are a task planning assistant. Given a user instruction, 
break it down into a structured plan with specific tasks.

Respond with a JSON object containing:
{
    "understanding": "Brief summary of what the user wants",
    "tasks": [
        {
            "name": "Task name",
            "agent_type": "browser|file|mcp",
            "description": "What this task does",
            "payload": { ... task-specific parameters ... },
            "dependencies": [] // indices of tasks this depends on
        }
    ]
}

Agent types:
- browser: Web browser actions (navigate, click, type, extract)
- file: Local file operations (read, write, list)
- mcp: External tool via MCP protocol"""

        messages = [Message(role="user", content=user_instruction)]
        if context:
            messages.insert(0, Message(role="user", content=f"Context: {context}"))

        response = await self.complete(
            messages=messages,
            system_prompt=system_prompt,
            temperature=0.3,
        )

        # Parse JSON from response
        try:
            # Try to extract JSON from the response
            content = response.content
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            # Return a simple task if parsing fails
            return {
                "understanding": user_instruction,
                "tasks": [{
                    "name": "Execute instruction",
                    "agent_type": "browser",
                    "description": user_instruction,
                    "payload": {"instruction": user_instruction},
                    "dependencies": [],
                }],
            }


# Global instance
llm_service = LLMService()

