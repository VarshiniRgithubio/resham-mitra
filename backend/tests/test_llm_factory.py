import asyncio

import pytest

from app.core.config import Settings
from app.llm.factory import build_llm


def test_fake_provider_works_offline():
    llm = build_llm(Settings(llm_provider="fake"))
    assert asyncio.run(llm.generate("hello")).startswith("[FAKE_LLM_REPLY]")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        build_llm(Settings(llm_provider="nope"))


def test_real_provider_needs_key():
    with pytest.raises(ValueError):
        build_llm(Settings(llm_provider="openai", llm_model="", llm_api_key=""))
