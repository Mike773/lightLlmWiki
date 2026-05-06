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
        lookup = inputs.llm.complete_json(expansion_prompt, ExpansionLookup)
        update_stage_entity(
            inputs.conn,
            stage_id,
            description=lookup.expansion,
            related_entity_ids=[e.id for e in matches],
        )

    inputs.conn.commit()
