import json
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    def complete_text(self, prompt: str) -> str: ...

    def complete_json(self, prompt: str, schema: type[T]) -> T: ...


_JSON_INSTRUCTION = (
    "Верни ответ строго в формате JSON, соответствующем схеме ниже. "
    "Никаких пояснений, никакого markdown, никаких комментариев — только сам JSON-объект.\n\n"
    "JSON Schema:\n{schema_json}"
)


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def complete_text(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def complete_json(self, prompt: str, schema: type[T]) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        full_prompt = f"{prompt}\n\n{_JSON_INSTRUCTION.format(schema_json=schema_json)}"
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return schema.model_validate_json(content)
