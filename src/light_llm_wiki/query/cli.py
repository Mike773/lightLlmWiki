import argparse
import os
import sys

from light_llm_wiki.config import load_user_config
from light_llm_wiki.db import get_connection
from light_llm_wiki.embedding import FunctionEmbeddingClient, OpenAIEmbeddingClient
from light_llm_wiki.llm import FunctionLLMClient, OpenAIChatClient
from light_llm_wiki.query.runner import answer_question

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/light_llm_wiki"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="light-llm-wiki-query",
        description=(
            "Ask a natural-language question against the knowledge base "
            "for a given direction."
        ),
    )
    parser.add_argument(
        "--direction",
        required=True,
        help="direction key (must exist in llm_wiki_rag.directions)",
    )
    parser.add_argument(
        "--question",
        help="the question text; if omitted, it is read from stdin",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.5,
        help="cosine distance threshold for embedding-fallback resolution (default: 0.5)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DSN", DEFAULT_DSN),
        help=f"PostgreSQL DSN (default: env DSN or {DEFAULT_DSN})",
    )
    parser.add_argument(
        "--chat-model",
        default=os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
        help="OpenAI chat model (default: env CHAT_MODEL or gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large"),
        help="OpenAI embedding model (default: env EMBEDDING_MODEL or text-embedding-3-large)",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=int(os.environ.get("EMBEDDING_DIMENSIONS", "2560")),
        help="Embedding vector size; must match vector(N) in schema (default: 2560)",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible endpoint URL (default: env OPENAI_BASE_URL or None)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("LLM_MAX_RETRIES", "3")),
        help="LLM JSON parsing retries on schema-validation failure (default: 3)",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        default=os.environ.get("LLM_JSON_MODE", "").lower() in {"1", "true", "yes"},
        help=(
            "Send response_format={'type':'json_object'} to the chat API. "
            "Off by default; enable only if your provider supports JSON mode."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.question is not None:
        question = args.question
    else:
        question = sys.stdin.read().strip()
    if not question:
        sys.exit("question is empty")

    cfg = load_user_config()
    if cfg is not None:
        dsn = getattr(cfg, "dsn", None) or args.dsn
        llm = FunctionLLMClient(cfg.get_llm(), max_retries=args.max_retries)
        if not hasattr(cfg, "get_embeddings"):
            sys.exit(
                "lightllm_config.py must define get_embeddings() for the query CLI "
                "(it is required for embedding-fallback resolution)."
            )
        embedder = FunctionEmbeddingClient(cfg.get_embeddings())
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        dsn = args.dsn
        llm = OpenAIChatClient(
            model=args.chat_model,
            base_url=args.openai_base_url,
            api_key=api_key,
            max_retries=args.max_retries,
            use_json_mode=args.json_mode,
        )
        embedder = OpenAIEmbeddingClient(
            model=args.embedding_model,
            dimensions=args.embedding_dimensions,
            base_url=args.openai_base_url,
            api_key=api_key,
        )

    conn = get_connection(dsn)
    try:
        result = answer_question(
            conn,
            llm,
            embedder,
            args.direction,
            question,
            embedding_threshold=args.embedding_threshold,
        )
        sys.stdout.write(result.trace)
    finally:
        conn.close()
