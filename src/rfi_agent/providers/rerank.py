from __future__ import annotations

from dataclasses import dataclass

import httpx

from rfi_agent.config import Settings, get_settings
from rfi_agent.providers.factory import ProviderNotConfigured


@dataclass
class HttpReranker:
    provider: str
    model: str
    api_key: str
    base_url: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        if not documents:
            return []
        name = self.provider.strip().lower()
        try:
            if name == "cohere":
                return self._cohere(query, documents, top_n)
            return self._jina_or_openai(query, documents, top_n)
        except (httpx.HTTPError, KeyError, ValueError):
            return list(range(min(top_n, len(documents))))

    def _jina_or_openai(self, query: str, documents: list[str], top_n: int) -> list[int]:
        url = self.base_url.rstrip("/") + "/rerank"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "query": query, "documents": documents, "top_n": top_n}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results") or data.get("data") or []
        order = []
        for item in results:
            idx = item.get("index")
            if idx is None and isinstance(item, dict):
                idx = item.get("document_index")
            if isinstance(idx, int) and 0 <= idx < len(documents):
                order.append(idx)
        if not order:
            return list(range(min(top_n, len(documents))))
        return order[:top_n]

    def _cohere(self, query: str, documents: list[str], top_n: int) -> list[int]:
        url = (self.base_url or "https://api.cohere.com/v2").rstrip("/") + "/rerank"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "query": query, "documents": documents, "top_n": top_n}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results") or []
        order = [int(item["index"]) for item in results if "index" in item]
        return order[:top_n] if order else list(range(min(top_n, len(documents))))


def get_reranker(settings: Settings | None = None) -> HttpReranker | None:
    settings = settings or get_settings()
    name = settings.rerank_provider.strip().lower()
    if not name:
        return None
    key = settings.resolve_rerank_key()
    model = settings.rerank_model.strip()
    if name == "cohere":
        base = settings.rerank_base_url or "https://api.cohere.com/v2"
        model = model or "rerank-v3.5"
    elif name == "jina":
        base = settings.rerank_base_url or "https://api.jina.ai/v1"
        model = model or "jina-reranker-v2-base-multilingual"
    elif name == "openai_compatible":
        base = settings.rerank_base_url or settings.llm_base_url or "https://api.openai.com/v1"
        if not model:
            raise ProviderNotConfigured("Set RERANK_MODEL for openai_compatible rerank.")
    else:
        raise ProviderNotConfigured(f"Unknown RERANK_PROVIDER={settings.rerank_provider}. Use jina, cohere, or openai_compatible.")
    if not key and name != "openai_compatible":
        raise ProviderNotConfigured("Set RERANK_API_KEY for the rerank provider.")
    return HttpReranker(provider=name, model=model, api_key=key or "empty", base_url=base)
