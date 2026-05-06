from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...


class FunctionEmbeddingClient:
    """Adapt a plain (text: str) -> list[float] callable to EmbeddingClient.

    Used by the CLI to wrap user-provided `get_embeddings()` from
    `lightllm_config.py`.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]]) -> None:
        self._fn = embed_fn

    def embed(self, text: str) -> list[float]:
        return self._fn(text)


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        dimensions: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, object] = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return list(response.data[0].embedding)
