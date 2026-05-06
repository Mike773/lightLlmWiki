import json
import re
from typing import Callable, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

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

_RETRY_INSTRUCTION = (
    "Твой предыдущий ответ не прошёл валидацию схемы. Ошибка валидатора:\n"
    "{error}\n\n"
    "Верни строго валидный JSON по той же схеме, без markdown, без префикса "
    "и без пояснений — только сам JSON."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _extract_json(text: str) -> str:
    """Pull a JSON payload out of a possibly-noisy LLM response.

    Tries, in order: a fenced code block (``` or ```json), then the substring
    between the first opening and last matching closing brace/bracket. Falls
    back to the original text so Pydantic surfaces a clear error.
    """
    if not text:
        return text

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        end = text.rfind(close_ch)
        if end > start:
            return text[start : end + 1].strip()

    return text.strip()


def _parse_json_to_schema(json_text: str, schema: type[T]) -> T:
    """Validate `json_text` against `schema` with progressively lenient parsers.

    Order of attempts:
    1. Pydantic's strict JSON validator — handles well-formed JSON quickly.
    2. stdlib json.loads(strict=False) — accepts unescaped control
       characters (\\x00-\\x1f) inside string values, which models sometimes
       emit when echoing source text. Result is fed to model_validate.
    3. Strip control characters and re-run Pydantic's validator.

    Any remaining ValidationError / JSONDecodeError propagates so the
    caller can put it into the retry feedback loop.
    """
    try:
        return schema.model_validate_json(json_text)
    except (ValidationError, json.JSONDecodeError, ValueError):
        pass

    try:
        data = json.loads(json_text, strict=False)
        return schema.model_validate(data)
    except (ValidationError, json.JSONDecodeError, ValueError):
        pass

    cleaned = _CONTROL_CHARS_RE.sub("", json_text)
    return schema.model_validate_json(cleaned)


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        use_json_mode: bool = False,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._use_json_mode = use_json_mode

    def complete_text(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def complete_json(self, prompt: str, schema: type[T]) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        full_prompt = f"{prompt}\n\n{_JSON_INSTRUCTION.format(schema_json=schema_json)}"

        messages: list[dict[str, str]] = [{"role": "user", "content": full_prompt}]
        last_error: Exception | None = None
        last_content: str = ""

        for attempt in range(self._max_retries + 1):
            kwargs: dict[str, object] = {
                "model": self._model,
                "messages": messages,
            }
            if self._use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            last_content = content
            json_text = _extract_json(content)

            try:
                return _parse_json_to_schema(json_text, schema)
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt == self._max_retries:
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": _RETRY_INSTRUCTION.format(error=str(e)),
                    }
                )

        snippet = last_content[:500].rstrip()
        raise RuntimeError(
            f"complete_json: failed to obtain valid JSON for {schema.__name__} "
            f"after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}\nLast response (truncated): {snippet}"
        )


class FunctionLLMClient:
    """Adapt a plain (prompt: str) -> str callable to the LLMClient protocol.

    Used by the CLI to wrap user-provided `get_llm()` from
    `lightllm_config.py`. The wrapped callable does one chat completion;
    the wrapper applies the same retry + JSON-extraction loop as
    OpenAIChatClient, single-turn (the validator error is appended to
    the prompt for the next attempt rather than sent as a separate
    chat message, since the callable has no notion of message history).
    """

    def __init__(
        self,
        complete_fn: Callable[[str], str],
        *,
        max_retries: int = 3,
    ) -> None:
        self._fn = complete_fn
        self._max_retries = max_retries

    def complete_text(self, prompt: str) -> str:
        return self._fn(prompt)

    def complete_json(self, prompt: str, schema: type[T]) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        convo = f"{prompt}\n\n{_JSON_INSTRUCTION.format(schema_json=schema_json)}"

        last_error: Exception | None = None
        last_content: str = ""

        for attempt in range(self._max_retries + 1):
            content = self._fn(convo)
            last_content = content
            json_text = _extract_json(content)

            try:
                return _parse_json_to_schema(json_text, schema)
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt == self._max_retries:
                    break
                convo = (
                    f"{convo}\n\nПредыдущий ответ:\n{content}\n\n"
                    f"{_RETRY_INSTRUCTION.format(error=str(e))}"
                )

        snippet = last_content[:500].rstrip()
        raise RuntimeError(
            f"complete_json: failed to obtain valid JSON for {schema.__name__} "
            f"after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}\nLast response (truncated): {snippet}"
        )
