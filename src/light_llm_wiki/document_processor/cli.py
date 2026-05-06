import argparse
import os

from light_llm_wiki.config import load_user_config
from light_llm_wiki.db import get_connection
from light_llm_wiki.document_processor.pipeline import StageInputs, run_stage
from light_llm_wiki.embedding import OpenAIEmbeddingClient
from light_llm_wiki.llm import OpenAIChatClient

ALL_STAGES = ("abbreviations", "entities", "relations", "promote", "wiki")
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/light_llm_wiki"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="light-llm-wiki-pipeline",
        description=(
            "Run the document_processor pipeline against a document already "
            "loaded into llm_wiki_rag.documents."
        ),
    )
    parser.add_argument(
        "--document-id",
        required=True,
        type=int,
        help="id of a row in llm_wiki_rag.documents",
    )
    parser.add_argument(
        "--stage",
        choices=(*ALL_STAGES, "all"),
        default="all",
        help=(
            f"stage to run; one of {', '.join(ALL_STAGES)}, or 'all' "
            "to run every stage in order (default: all)"
        ),
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

    stages_to_run: tuple[str, ...]
    if args.stage == "all":
        stages_to_run = ALL_STAGES
    else:
        stages_to_run = (args.stage,)

    cfg = load_user_config()
    if cfg is not None:
        dsn = getattr(cfg, "dsn", None) or args.dsn
        llm = cfg.get_llm()
        embedder = cfg.get_embeddings() if hasattr(cfg, "get_embeddings") else None
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
        inputs = StageInputs(
            conn=conn, llm=llm, embedder=embedder, document_id=args.document_id
        )
        for stage in stages_to_run:
            print(f"=== running stage {stage!r} on document {args.document_id} ===")
            run_stage(stage, inputs)
        print("done")
    finally:
        conn.close()
