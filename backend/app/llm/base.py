"""Provider-agnostic LLM interface.

The rest of the code only talks to `LLMClient`, so switching OpenAI <-> Gemini
<-> Claude is a config change. This is also what lets us run the same
experiment across models later.
"""
from abc import ABC, abstractmethod


class LLMClient(ABC):
    name: str = "base"

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None,
                       temperature: float | None = None) -> str:
        """Return the model's text reply."""
