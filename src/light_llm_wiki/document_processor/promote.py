from light_llm_wiki.db import (
    find_entity_relation_by_pair,
    get_document,
    get_entity_by_name,
    insert_entity,
    insert_entity_relation,
    list_stage_entities_full,
    list_stage_relations,
    update_entity,
    update_entity_relation,
)
from light_llm_wiki.document_processor.pipeline import StageInputs


def promote_to_main(inputs: StageInputs) -> None:
    doc = get_document(inputs.conn, inputs.document_id)
    direction_key = doc.direction_key

    stage_entity_to_entity_id: dict[int, int] = {}

    for stage in list_stage_entities_full(inputs.conn):
        is_abbreviation = stage.type == "abbreviation"
        has_expansion = is_abbreviation and stage.description is not None

        existing = get_entity_by_name(
            inputs.conn, direction_key, stage.type, stage.name
        )
        if existing is None:
            entity_id = insert_entity(
                inputs.conn,
                direction_key=direction_key,
                type=stage.type,
                name=stage.name,
                description=stage.description,
                is_abbreviation=is_abbreviation,
                has_expansion=has_expansion,
                embedding=stage.embedding,
            )
        else:
            update_entity(
                inputs.conn,
                existing.id,
                description=stage.description,
                is_abbreviation=is_abbreviation,
                has_expansion=has_expansion,
                embedding=stage.embedding,
            )
            entity_id = existing.id

        stage_entity_to_entity_id[stage.id] = entity_id

    for rel in list_stage_relations(inputs.conn):
        source_entity_id = stage_entity_to_entity_id.get(rel.source_stage_id)
        target_entity_id = stage_entity_to_entity_id.get(rel.target_stage_id)
        if source_entity_id is None or target_entity_id is None:
            continue

        existing = find_entity_relation_by_pair(
            inputs.conn, direction_key, source_entity_id, target_entity_id
        )
        if existing is None:
            insert_entity_relation(
                inputs.conn,
                direction_key=direction_key,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relation_type=rel.relation_type,
                description=rel.description,
                document_id=doc.id,
                quote=rel.quote,
            )
        else:
            update_entity_relation(
                inputs.conn,
                existing.id,
                description=rel.description,
                document_id=doc.id,
                quote=rel.quote,
            )

    inputs.conn.commit()
