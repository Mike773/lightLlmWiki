from light_llm_wiki.db import (
    clear_stage_by_type,
    find_similar_entities,
    get_document,
    insert_stage_entity,
    list_stage_abbreviations,
    update_stage_entity,
)
from light_llm_wiki.document_processor import abbreviation_block
from light_llm_wiki.document_processor.pipeline import StageInputs
from light_llm_wiki.document_processor.prompts import (
    ENTITIES_PROMPT,
    ENTITY_DESCRIPTION_PROMPT,
)
from light_llm_wiki.document_processor.schemas import (
    EntitiesList,
    EntityDescription,
)


def extract_entities(inputs: StageInputs) -> None:
    if inputs.embedder is None:
        raise ValueError("entities stage requires an embedder")

    doc = get_document(inputs.conn, inputs.document_id)
    abbreviations = list_stage_abbreviations(inputs.conn)
    abbrev_block = abbreviation_block.format_for_prompt(abbreviations)
    abbrev_exclude_block = abbreviation_block.format_exclude_list(abbreviations)

    names_prompt = ENTITIES_PROMPT.format(
        content=doc.content,
        abbreviations_to_exclude=abbrev_exclude_block,
    )
    extraction = inputs.llm.complete_json(names_prompt, EntitiesList)

    clear_stage_by_type(inputs.conn, "entity")

    stage_items: list[tuple[int, str]] = []
    for name in extraction.items:
        stage_id = insert_stage_entity(
            inputs.conn,
            type="entity",
            name=name,
            description=None,
            related_entity_ids=[],
            embedding=None,
        )
        stage_items.append((stage_id, name))

    for stage_id, name in stage_items:
        desc_prompt = ENTITY_DESCRIPTION_PROMPT.format(
            content=doc.content,
            name=name,
            abbreviations_block=abbrev_block,
        )
        try:
            lookup = inputs.llm.complete_json(desc_prompt, EntityDescription)
            description = lookup.description
        except Exception as e:
            print(
                f"[entities] LLM failed on description of {name!r}: {e}; "
                "leaving description empty"
            )
            description = None

        embed_text = name if description is None else f"{name}\n\n{description}"
        try:
            emb = inputs.embedder.embed(embed_text)
        except Exception as e:
            print(
                f"[entities] embedder failed on {name!r}: {e}; "
                "leaving embedding empty"
            )
            emb = None

        if emb is not None:
            related = find_similar_entities(
                inputs.conn, doc.direction_key, emb, limit=10
            )
        else:
            related = []

        update_stage_entity(
            inputs.conn,
            stage_id,
            description=description,
            related_entity_ids=[e.id for e in related],
            embedding=emb,
        )

    inputs.conn.commit()
