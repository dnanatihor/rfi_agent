from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from rfi_agent.config import Settings, get_settings


class ProviderNotConfigured(RuntimeError):
    pass


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    name = settings.llm_provider.strip().lower()
    if name == "openai_compatible":
        from langchain_openai import ChatOpenAI

        if not settings.resolve_llm_key() and not (settings.llm_base_url or "").strip():
            raise ProviderNotConfigured("Set LLM_API_KEY (or LLM_BASE_URL for an OpenAI-compatible server).")
        kwargs: dict = {"model": settings.llm_model, "temperature": 0, "api_key": settings.resolve_llm_key() or "empty"}
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        return ChatOpenAI(**kwargs)
    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.resolve_llm_key():
            raise ProviderNotConfigured("Set LLM_API_KEY for Anthropic.")
        return ChatAnthropic(model=settings.llm_model, api_key=settings.resolve_llm_key(), temperature=0)
    if name == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(model=settings.llm_model, region_name=settings.aws_region)
    raise ProviderNotConfigured(
        f"Unknown LLM_PROVIDER={settings.llm_provider}. Use openai_compatible, anthropic, or bedrock."
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    name = settings.embedding_provider.strip().lower()
    if name == "openai_compatible":
        from langchain_openai import OpenAIEmbeddings

        key = settings.resolve_embedding_key()
        if not key and not (settings.embedding_base_url or settings.llm_base_url or "").strip():
            raise ProviderNotConfigured("Set EMBEDDING_API_KEY or LLM_API_KEY for embeddings.")
        kwargs: dict = {"model": settings.embedding_model, "api_key": key or "empty"}
        base = settings.embedding_base_url or settings.llm_base_url
        if base:
            kwargs["base_url"] = base
        return OpenAIEmbeddings(**kwargs)
    if name == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=settings.embedding_model, region_name=settings.aws_region)
    raise ProviderNotConfigured(
        f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider}. Use openai_compatible or bedrock."
    )
