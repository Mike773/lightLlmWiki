from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...
