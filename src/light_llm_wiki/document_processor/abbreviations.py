from light_llm_wiki.db import (
    clear_stage,
    find_abbreviation_entities_by_name,
    get_document,
    insert_stage_entity,
)
from light_llm_wiki.document_processor.pipeline import StageInputs
from light_llm_wiki.document_processor.prompts import ABBREVIATIONS_PROMPT
from light_llm_wiki.document_processor.schemas import AbbreviationsExtraction


def extract_abbreviations(inputs: StageInputs) -> None:
    doc = get_document(inputs.conn, inputs.document_id)

    prompt = ABBREVIATIONS_PROMPT.format(content=doc.content)
    extraction = inputs.llm.complete_json(prompt, AbbreviationsExtraction)

    clear_stage(inputs.conn)
    for item in extraction.items:
        matches = find_abbreviation_entities_by_name(
            inputs.conn, doc.direction_key, item.name
        )
        insert_stage_entity(
            inputs.conn,
            type="abbreviation",
            name=item.name,
            description=item.expansion,
            related_entity_ids=[e.id for e in matches],
        )

    inputs.conn.commit()
