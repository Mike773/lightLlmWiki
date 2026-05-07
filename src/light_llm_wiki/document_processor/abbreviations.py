import re

from light_llm_wiki.db import (
    clear_stage,
    find_abbreviation_entities_by_name,
    get_document,
    insert_stage_entity,
    update_stage_entity,
)
from light_llm_wiki.document_processor.pipeline import StageInputs
from light_llm_wiki.document_processor.prompts import (
    ABBREVIATIONS_PROMPT,
    EXPANSION_PROMPT,
)
from light_llm_wiki.document_processor.schemas import (
    AbbreviationsList,
    ExpansionLookup,
)


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_substring(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def _verify_expansion(
    name: str, lookup: ExpansionLookup, content: str
) -> str | None:
    if lookup.expansion is None:
        return None
    if lookup.quote is None:
        print(
            f"[abbreviations] rejected hallucinated expansion for {name!r}: "
            "quote not provided"
        )
        return None
    if _normalize_for_substring(lookup.quote) not in _normalize_for_substring(content):
        print(
            f"[abbreviations] rejected hallucinated expansion for {name!r}: "
            "quote not found in document"
        )
        return None
    return lookup.expansion


def extract_abbreviations(inputs: StageInputs) -> None:
    doc = get_document(inputs.conn, inputs.document_id)

    names_prompt = ABBREVIATIONS_PROMPT.format(content=doc.content)
    extraction = inputs.llm.complete_json(names_prompt, AbbreviationsList)

    clear_stage(inputs.conn)
    stage_items: list[tuple[int, str]] = []
    for name in extraction.items:
        stage_id = insert_stage_entity(
            inputs.conn,
            type="abbreviation",
            name=name,
            description=None,
            related_entity_ids=[],
        )
        stage_items.append((stage_id, name))

    for stage_id, name in stage_items:
        matches = find_abbreviation_entities_by_name(
            inputs.conn, doc.direction_key, name
        )
        expansion_prompt = EXPANSION_PROMPT.format(content=doc.content, name=name)
        try:
            lookup = inputs.llm.complete_json(expansion_prompt, ExpansionLookup)
        except Exception as e:
            print(
                f"[abbreviations] LLM failed on expansion of {name!r}: {e}; "
                "leaving description empty"
            )
            description = None
        else:
            description = _verify_expansion(name, lookup, doc.content)
        update_stage_entity(
            inputs.conn,
            stage_id,
            description=description,
            related_entity_ids=[e.id for e in matches],
        )

    inputs.conn.commit()
