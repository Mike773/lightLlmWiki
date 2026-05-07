from light_llm_wiki.db import (
    StageEntity,
    clear_stage_relations,
    delete_stage_relation,
    get_document,
    insert_stage_relation,
    list_stage_abbreviations,
    list_stage_entities,
    update_stage_relation,
)
from light_llm_wiki.document_processor import abbreviation_block
from light_llm_wiki.document_processor.pipeline import StageInputs
from light_llm_wiki.document_processor.prompts import (
    RELATION_DESCRIPTION_PROMPT,
    RELATIONS_PROMPT,
)
from light_llm_wiki.document_processor.schemas import (
    RelationCandidatesList,
    RelationLookup,
)


def _format_entities_block(entities: list[StageEntity]) -> str:
    lines = []
    for ent in entities:
        desc = ent.description if ent.description else "—"
        lines.append(f"- id={ent.id} — {ent.name} — {desc}")
    return "\n".join(lines)


def extract_relations(inputs: StageInputs) -> None:
    doc = get_document(inputs.conn, inputs.document_id)
    entities = list_stage_entities(inputs.conn)
    if len(entities) < 2:
        clear_stage_relations(inputs.conn)
        inputs.conn.commit()
        return

    by_id: dict[int, StageEntity] = {e.id: e for e in entities}
    valid_ids = set(by_id)
    entities_block = _format_entities_block(entities)
    abbrev_block = abbreviation_block.format_for_prompt(
        list_stage_abbreviations(inputs.conn)
    )

    clear_stage_relations(inputs.conn)

    seen_pairs: set[frozenset[int]] = set()
    inserted: list[tuple[int, StageEntity, StageEntity, str]] = []

    for source in entities:
        prompt = RELATIONS_PROMPT.format(
            content=doc.content,
            source_id=source.id,
            source_name=source.name,
            entities_block=entities_block,
            abbreviations_block=abbrev_block,
        )
        try:
            candidates = inputs.llm.complete_json(prompt, RelationCandidatesList)
        except Exception as e:
            print(
                f"[relations] LLM failed on candidates for {source.name!r}: {e}; "
                "skipping its relations"
            )
            continue

        for cand in candidates.items:
            target_id = cand.target_id
            if target_id == source.id or target_id not in valid_ids:
                continue
            relation_type = cand.relation_type.strip().lower()
            if not relation_type:
                continue
            pair = frozenset({source.id, target_id})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            target = by_id[target_id]
            stage_id = insert_stage_relation(
                inputs.conn,
                source_stage_id=source.id,
                target_stage_id=target_id,
                relation_type=relation_type,
            )
            inserted.append((stage_id, source, target, relation_type))

    for stage_id, source, target, relation_type in inserted:
        prompt = RELATION_DESCRIPTION_PROMPT.format(
            content=doc.content,
            source_name=source.name,
            target_name=target.name,
            relation_type=relation_type,
            abbreviations_block=abbrev_block,
        )
        try:
            lookup = inputs.llm.complete_json(prompt, RelationLookup)
        except Exception as e:
            print(
                f"[relations] LLM failed on description "
                f"{source.name!r} -> {target.name!r}: {e}; dropping the relation"
            )
            delete_stage_relation(inputs.conn, stage_id)
            continue

        if lookup.quote is None:
            delete_stage_relation(inputs.conn, stage_id)
            continue

        update_stage_relation(
            inputs.conn,
            stage_id,
            description=lookup.description,
            quote=lookup.quote,
        )

    inputs.conn.commit()
