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
    from light_llm_wiki.document_processor.entities import extract_entities
    from light_llm_wiki.document_processor.promote import promote_to_main
    from light_llm_wiki.document_processor.relations import extract_relations

    return {
        "abbreviations": extract_abbreviations,
        "entities": extract_entities,
        "relations": extract_relations,
        "promote": promote_to_main,
    }


def run_stage(name: str, inputs: StageInputs) -> None:
    stages = _stages()
    if name not in stages:
        raise ValueError(f"unknown stage: {name!r}; available: {sorted(stages)}")
    stages[name](inputs)
