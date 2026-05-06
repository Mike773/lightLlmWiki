from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    def complete_text(self, prompt: str) -> str: ...

    def complete_json(self, prompt: str, schema: type[T]) -> T: ...
