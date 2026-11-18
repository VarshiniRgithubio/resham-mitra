"""Concrete LLM providers. SDKs are imported lazily so you only install the
one you actually use."""
from app.llm.base import LLMClient


class FakeLLM(LLMClient):
    """Offline, deterministic client for tests and local dev without a key."""
    name = "fake"

    async def generate(self, prompt, system=None, temperature=None):
        # In the chat flow, echo the top passage so the UI shows something
        # meaningful without an API key. Clearly labelled as not a real answer.
        if prompt.startswith("Context:\n[1]"):
            first = prompt.split("\n", 2)[1].split(") ", 1)[-1]
            return f"[FAKE_LLM_REPLY - no API key] Top passage [1]: {first[:200]}"
        return f"[FAKE_LLM_REPLY] {prompt[:80]}"


class OpenAILLM(LLMClient):
    name = "openai"

    def __init__(self, model: str, api_key: str, temperature: float):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.model, self.temperature = model, temperature

    async def generate(self, prompt, system=None, temperature=None):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature if temperature is None else temperature)
        return resp.choices[0].message.content or ""


class AnthropicLLM(LLMClient):
    name = "anthropic"

    def __init__(self, model: str, api_key: str, temperature: float):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self.model, self.temperature = model, temperature

    async def generate(self, prompt, system=None, temperature=None):
        kwargs = {"system": system} if system else {}
        resp = await self.client.messages.create(
            model=self.model, max_tokens=1024,
            temperature=self.temperature if temperature is None else temperature,
            messages=[{"role": "user", "content": prompt}], **kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")


class GeminiLLM(LLMClient):
    name = "gemini"

    def __init__(self, model: str, api_key: str, temperature: float):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai, self.model, self.temperature = genai, model, temperature

    async def generate(self, prompt, system=None, temperature=None):
        model = self.genai.GenerativeModel(self.model, system_instruction=system)
        resp = await model.generate_content_async(
            prompt, generation_config={
                "temperature": self.temperature if temperature is None else temperature})
        return resp.text


class ExtractiveLLM(LLMClient):
    """No AI at all. The pipeline shows the best-matching sentences from the
    sources instead of generating text, so nothing can be invented. Used as a
    zero-cost, zero-hallucination baseline in the experiments."""
    name = "extractive"

    async def generate(self, prompt, system=None, temperature=None):
        # Only reached for translation / judging, which need a real model.
        return prompt.rsplit("\n\n", 1)[-1]


class OllamaLLM(LLMClient):
    """Free local models via Ollama (https://ollama.com), fully offline."""
    name = "ollama"

    def __init__(self, model: str, base_url: str, temperature: float):
        import httpx
        self.http = httpx.AsyncClient(base_url=base_url, timeout=300)
        self.model, self.temperature = model, temperature

    async def generate(self, prompt, system=None, temperature=None):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        r = await self.http.post("/api/chat", json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature}})
        r.raise_for_status()
        return r.json()["message"]["content"]
