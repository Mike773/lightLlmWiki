"""
Local config for light-llm-wiki CLI.

Copy this file next to your working directory as `lightllm_config.py`
(it is gitignored) and fill in the DSN and the two factory functions.

If `lightllm_config.py` is present, the CLI uses it instead of the
env-based defaults. If it is absent, the CLI falls back to the
OPENAI_* env variables and the built-in OpenAIChatClient /
OpenAIEmbeddingClient — nothing else changes.
"""

# Optional. If omitted, the CLI uses --dsn / DSN env / built-in default.
dsn = "postgresql://postgres:postgres@localhost:5432/light_llm_wiki"


def get_llm():
    """Return any object that implements light_llm_wiki.llm.LLMClient."""
    from light_llm_wiki.llm import OpenAIChatClient

    return OpenAIChatClient(
        model="gpt-4o-mini",
        api_key="sk-...",
        # base_url="http://localhost:8080/v1",
        # max_retries=3,
        # use_json_mode=False,
    )


def get_embeddings():
    """Return any object that implements light_llm_wiki.embedding.EmbeddingClient.

    Optional: if you don't need the embeddings stages (entities, relations,
    promote, wiki, query) you can omit this function — the CLI passes None
    to stages, and stages that require embeddings raise a clear error.
    """
    from light_llm_wiki.embedding import OpenAIEmbeddingClient

    return OpenAIEmbeddingClient(
        model="text-embedding-3-large",
        dimensions=2560,
        api_key="sk-...",
        # base_url="http://localhost:8080/v1",
    )
