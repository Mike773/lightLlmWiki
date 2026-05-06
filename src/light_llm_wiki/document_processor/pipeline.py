from dataclasses import dataclass
from typing import Callable

import psycopg

from light_llm_wiki.embedding import EmbeddingClient
from light_llm_wiki.llm import LLMClient


@dataclass
class StageInputs:
    conn: psycopg.Connection
    llm: LLMClient
    embedder: EmbeddingClient | None
    document_id: int


StageFn = Callable[[StageInputs], None]


def _stages() -> dict[str, StageFn]:
    from light_llm_wiki.document_processor.abbreviations import extract_abbreviations

    return {
        "abbreviations": extract_abbreviations,
    }


def run_stage(name: str, inputs: StageInputs) -> None:
    stages = _stages()
    if name not in stages:
        raise ValueError(f"unknown stage: {name!r}; available: {sorted(stages)}")
    stages[name](inputs)
