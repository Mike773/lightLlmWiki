"""
Local config for light-llm-wiki CLI.

Copy this file next to your working directory as `lightllm_config.py`
(it is gitignored). The CLI auto-loads it and uses your `get_llm()` /
`get_embeddings()` instead of the built-in OpenAI clients.

`get_llm()` and `get_embeddings()` must return PLAIN CALLABLES, not
client objects:

    get_llm()        -> Callable[[str], str]
    get_embeddings() -> Callable[[str], list[float]]

The CLI wraps each callable with a thin adapter:
- the LLM callable is called by the retry-and-extract-JSON loop, so
  you only need to do a single chat completion;
- the embedding callable is called once per text.

If `lightllm_config.py` is absent, the CLI falls back to the env-based
flow (CHAT_MODEL, EMBEDDING_MODEL, OPENAI_API_KEY, …) — nothing else
changes.
"""

# Optional. If omitted, the CLI uses --dsn / DSN env / built-in default.
dsn = "postgresql://postgres:postgres@localhost:5432/light_llm_wiki"


def get_llm():
    """Return a callable (prompt: str) -> str for chat completion."""
    from openai import OpenAI

    client = OpenAI(api_key="sk-...")  # or base_url=... for a local endpoint

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return complete


def get_embeddings():
    """Return a callable (text: str) -> list[float].

    Optional: if you don't need the embedding-using stages
    (entities, relations, promote, wiki) and the query CLI, you can
    omit this function. The pipeline CLI passes None to stages, and
    stages that need embeddings raise a clear error.
    """
    from openai import OpenAI

    client = OpenAI(api_key="sk-...")

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=2560,
        )
        return list(response.data[0].embedding)

    return embed
