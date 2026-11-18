from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.providers import (AnthropicLLM, ExtractiveLLM, FakeLLM, GeminiLLM, OllamaLLM,
                               OpenAILLM)

_PROVIDERS = {"openai": OpenAILLM, "anthropic": AnthropicLLM, "gemini": GeminiLLM}


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "fake":
        return FakeLLM()
    if provider == "extractive":
        return ExtractiveLLM()
    if provider == "ollama":  # local, no key needed
        return OllamaLLM(settings.llm_model or "gemma3:12b", settings.llm_base_url,
                         settings.llm_temperature)
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
                         f"Use one of: fake, extractive, ollama, {', '.join(_PROVIDERS)}")
    if not settings.llm_model or not settings.llm_api_key:
        raise ValueError("LLM_MODEL and LLM_API_KEY must be set for a real provider")
    return _PROVIDERS[provider](settings.llm_model, settings.llm_api_key,
                                settings.llm_temperature)
