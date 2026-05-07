from light_llm_wiki.db import (
    Direction,
    Document,
    Entity,
    EntityRelation,
    find_wiki_page_for_direction,
    find_wiki_page_for_document,
    find_wiki_page_for_entity,
    get_direction,
    get_document,
    get_entity_by_name,
    insert_wiki_page,
    list_documents_by_direction,
    list_documents_for_entity,
    list_entities_by_direction,
    list_entities_for_document,
    list_entity_relations_by_direction,
    list_entity_relations_by_document,
    list_entity_relations_by_entity,
    list_stage_entities_full,
    update_wiki_page,
    update_wiki_page_related_ids,
)
from light_llm_wiki.document_processor import abbreviation_block
from light_llm_wiki.document_processor.pipeline import StageInputs
from light_llm_wiki.document_processor.prompts import (
    WIKI_DIRECTION_OVERVIEW_PROMPT,
    WIKI_ENTITY_SUMMARY_PROMPT,
)
from light_llm_wiki.document_processor.schemas import (
    WikiDirectionOverview,
    WikiEntitySummary,
)


def _entity_link(entity: Entity) -> str:
    return f"[{entity.name}](wiki/entity/{entity.id})"


def _document_link(doc: Document) -> str:
    return f"[{doc.title}](wiki/document/{doc.id})"


def _format_relation_line(
    rel: EntityRelation, neighbor: Entity, *, direction: str
) -> str:
    arrow = "→" if direction == "out" else "←"
    line = f"- {arrow} {_entity_link(neighbor)} — {rel.relation_type}"
    if rel.description:
        line += f"\n  - {rel.description}"
    if rel.quote:
        line += f"\n  > {rel.quote}"
    return line


def _build_entity_relations_block_for_prompt(
    entity_id: int,
    relations: list[EntityRelation],
    by_id: dict[int, Entity],
) -> str:
    if not relations:
        return "(связи не зафиксированы)"
    lines = []
    for rel in relations:
        if rel.source_entity_id == entity_id:
            other = by_id.get(rel.target_entity_id)
            arrow = "→"
        else:
            other = by_id.get(rel.source_entity_id)
            arrow = "←"
        if other is None:
            continue
        parts = [f"{arrow} {other.name} — {rel.relation_type}"]
        if rel.description:
            parts.append(f"  описание: {rel.description}")
        if rel.quote:
            parts.append(f"  цитата: {rel.quote}")
        lines.append("\n".join(parts))
    return "\n".join(lines) if lines else "(связи не зафиксированы)"


def _build_documents_block_for_prompt(documents: list[Document]) -> str:
    if not documents:
        return "(не упоминается в зафиксированных связях)"
    return "\n".join(f"- {d.title}" for d in documents)


def _build_entity_page_content(
    entity: Entity,
    summary: str,
    relations: list[EntityRelation],
    by_id: dict[int, Entity],
    documents: list[Document],
) -> str:
    parts = [f"# {entity.name}", "", summary]

    parts += ["", "## Описание", ""]
    parts.append(entity.description or "_описание не зафиксировано_")

    parts += ["", "## Тип", ""]
    type_label = entity.type
    if entity.is_abbreviation:
        type_label += " (аббревиатура)"
    parts.append(type_label)

    parts += ["", "## Связи", ""]
    if not relations:
        parts.append("_связи не зафиксированы_")
    else:
        for rel in relations:
            if rel.source_entity_id == entity.id:
                other = by_id.get(rel.target_entity_id)
                direction = "out"
            else:
                other = by_id.get(rel.source_entity_id)
                direction = "in"
            if other is None:
                continue
            parts.append(_format_relation_line(rel, other, direction=direction))

    parts += ["", "## Упоминается в документах", ""]
    if not documents:
        parts.append("_не упоминается в зафиксированных связях_")
    else:
        for d in documents:
            parts.append(f"- {_document_link(d)}")

    return "\n".join(parts) + "\n"


def _build_document_page_content(
    doc: Document,
    relations: list[EntityRelation],
    by_id: dict[int, Entity],
    entity_ids: list[int],
) -> str:
    parts = [f"# {doc.title}", "", "## Метаданные", ""]
    parts.append(f"- ID документа: {doc.id}")
    parts.append(f"- Статус: {doc.status}")
    parts.append(f"- Использовать чанкинг: {doc.use_chunking}")

    parts += ["", "## Извлечённые сущности", ""]
    if not entity_ids:
        parts.append("_сущности не извлечены_")
    else:
        for eid in entity_ids:
            ent = by_id.get(eid)
            if ent is None:
                continue
            short = ent.description or ""
            short = short.strip().split("\n", 1)[0]
            if len(short) > 160:
                short = short[:157] + "..."
            line = f"- {_entity_link(ent)}"
            if short:
                line += f" — {short}"
            parts.append(line)

    parts += ["", "## Связи в этом документе", ""]
    if not relations:
        parts.append("_связи не зафиксированы_")
    else:
        for rel in relations:
            src = by_id.get(rel.source_entity_id)
            tgt = by_id.get(rel.target_entity_id)
            if src is None or tgt is None:
                continue
            line = (
                f"- {_entity_link(src)} → {_entity_link(tgt)} "
                f"— {rel.relation_type}"
            )
            if rel.description:
                line += f"\n  - {rel.description}"
            if rel.quote:
                line += f"\n  > {rel.quote}"
            parts.append(line)

    return "\n".join(parts) + "\n"


def _build_direction_page_content(
    direction: Direction,
    overview: str,
    entities: list[Entity],
    documents: list[Document],
) -> str:
    parts = [f"# {direction.name}"]
    if direction.description:
        parts += ["", direction.description]

    parts += ["", "## Обзор", "", overview]

    abbreviations = [e for e in entities if e.is_abbreviation]
    others = [e for e in entities if not e.is_abbreviation]

    parts += ["", "## Сущности", ""]
    if abbreviations:
        parts += ["### Аббревиатуры", ""]
        for ent in abbreviations:
            short = (ent.description or "").strip().split("\n", 1)[0]
            line = f"- {_entity_link(ent)}"
            if short:
                line += f" — {short}"
            parts.append(line)
        parts.append("")
    if others:
        parts += ["### Предметные сущности", ""]
        for ent in others:
            short = (ent.description or "").strip().split("\n", 1)[0]
            if len(short) > 160:
                short = short[:157] + "..."
            line = f"- {_entity_link(ent)}"
            if short:
                line += f" — {short}"
            parts.append(line)
        parts.append("")

    parts += ["## Документы", ""]
    if documents:
        for d in documents:
            parts.append(f"- {_document_link(d)}")
    else:
        parts.append("_документы не загружены_")

    return "\n".join(parts) + "\n"


def _resolve_related_page_ids(
    conn,
    direction_key: str,
    entity_ids: list[int],
    document_ids: list[int],
) -> list[int]:
    page_ids: list[int] = []
    for eid in entity_ids:
        page = find_wiki_page_for_entity(conn, direction_key, eid)
        if page is not None:
            page_ids.append(page.id)
    for did in document_ids:
        page = find_wiki_page_for_document(conn, direction_key, did)
        if page is not None:
            page_ids.append(page.id)
    return page_ids


def _upsert_entity_page(
    inputs: StageInputs,
    direction_key: str,
    entity: Entity,
    by_id: dict[int, Entity],
    docs_by_id: dict[int, Document],
    abbreviations_block: str,
) -> None:
    relations = list_entity_relations_by_entity(inputs.conn, entity.id)

    doc_ids: list[int] = []
    for rel in relations:
        if rel.document_id is not None and rel.document_id not in doc_ids:
            doc_ids.append(rel.document_id)
    for d in list_documents_for_entity(inputs.conn, entity.id):
        if d.id not in doc_ids:
            doc_ids.append(d.id)
    documents = [docs_by_id[d] for d in doc_ids if d in docs_by_id]

    relations_block = _build_entity_relations_block_for_prompt(
        entity.id, relations, by_id
    )
    documents_block = _build_documents_block_for_prompt(documents)

    prompt = WIKI_ENTITY_SUMMARY_PROMPT.format(
        name=entity.name,
        type=entity.type,
        description=entity.description or "(описание не зафиксировано)",
        abbreviations_block=abbreviations_block,
        relations_block=relations_block,
        documents_block=documents_block,
    )
    try:
        result = inputs.llm.complete_json(prompt, WikiEntitySummary)
        summary = result.summary.strip()
    except Exception as e:
        print(
            f"[wiki] LLM failed on summary of {entity.name!r}: {e}; "
            "leaving summary empty"
        )
        summary = ""

    content = _build_entity_page_content(entity, summary, relations, by_id, documents)

    embed_text = "\n\n".join(
        s for s in (entity.name, summary, entity.description or "") if s
    )
    embedding = inputs.embedder.embed(embed_text)

    neighbor_ids: list[int] = []
    for rel in relations:
        other = (
            rel.target_entity_id
            if rel.source_entity_id == entity.id
            else rel.source_entity_id
        )
        if other not in neighbor_ids:
            neighbor_ids.append(other)

    related_page_ids = _resolve_related_page_ids(
        inputs.conn, direction_key, neighbor_ids, doc_ids
    )
    relation_ids = [r.id for r in relations]

    existing = find_wiki_page_for_entity(inputs.conn, direction_key, entity.id)
    if existing is None:
        insert_wiki_page(
            inputs.conn,
            direction_key=direction_key,
            type="entity",
            entity_id=entity.id,
            document_id=None,
            title=entity.name,
            content=content,
            related_page_ids=related_page_ids,
            entity_ids=[entity.id],
            relation_ids=relation_ids,
            embedding=embedding,
        )
    else:
        update_wiki_page(
            inputs.conn,
            existing.id,
            title=entity.name,
            content=content,
            related_page_ids=related_page_ids,
            entity_ids=[entity.id],
            relation_ids=relation_ids,
            embedding=embedding,
        )


def _upsert_document_page(
    inputs: StageInputs,
    direction_key: str,
    doc: Document,
    by_id: dict[int, Entity],
) -> None:
    relations = list_entity_relations_by_document(inputs.conn, doc.id)

    entity_ids: list[int] = []
    for rel in relations:
        for eid in (rel.source_entity_id, rel.target_entity_id):
            if eid not in entity_ids:
                entity_ids.append(eid)
    for e in list_entities_for_document(inputs.conn, doc.id):
        if e.id not in entity_ids:
            entity_ids.append(e.id)

    content = _build_document_page_content(doc, relations, by_id, entity_ids)

    embed_text_parts = [doc.title, doc.content[:4000]]
    embed_text = "\n\n".join(s for s in embed_text_parts if s)
    embedding = inputs.embedder.embed(embed_text)

    related_page_ids = _resolve_related_page_ids(
        inputs.conn, direction_key, entity_ids, []
    )

    existing = find_wiki_page_for_document(inputs.conn, direction_key, doc.id)
    if existing is None:
        insert_wiki_page(
            inputs.conn,
            direction_key=direction_key,
            type="document",
            entity_id=None,
            document_id=doc.id,
            title=doc.title,
            content=content,
            related_page_ids=related_page_ids,
            entity_ids=entity_ids,
            relation_ids=[r.id for r in relations],
            embedding=embedding,
        )
    else:
        update_wiki_page(
            inputs.conn,
            existing.id,
            title=doc.title,
            content=content,
            related_page_ids=related_page_ids,
            entity_ids=entity_ids,
            relation_ids=[r.id for r in relations],
            embedding=embedding,
        )


def _upsert_direction_page(
    inputs: StageInputs,
    direction: Direction,
    entities: list[Entity],
    documents: list[Document],
    relations: list[EntityRelation],
    by_id: dict[int, Entity],
    abbreviations_block: str,
) -> None:
    entities_block_lines = []
    for ent in entities:
        desc = (ent.description or "").strip().split("\n", 1)[0]
        if len(desc) > 200:
            desc = desc[:197] + "..."
        entities_block_lines.append(f"- {ent.name} ({ent.type}) — {desc or '—'}")
    entities_block = "\n".join(entities_block_lines) if entities_block_lines else "(сущностей нет)"

    relations_block_lines = []
    for rel in relations:
        src = by_id.get(rel.source_entity_id)
        tgt = by_id.get(rel.target_entity_id)
        if src is None or tgt is None:
            continue
        line = f"- {src.name} → {tgt.name} — {rel.relation_type}"
        if rel.description:
            line += f": {rel.description}"
        relations_block_lines.append(line)
    relations_block = "\n".join(relations_block_lines) if relations_block_lines else "(связей нет)"

    documents_block_lines = [f"- {d.title}" for d in documents]
    documents_block = "\n".join(documents_block_lines) if documents_block_lines else "(документов нет)"

    prompt = WIKI_DIRECTION_OVERVIEW_PROMPT.format(
        direction_name=direction.name,
        direction_description=direction.description or "(описание не задано)",
        abbreviations_block=abbreviations_block,
        entities_block=entities_block,
        relations_block=relations_block,
        documents_block=documents_block,
    )
    try:
        result = inputs.llm.complete_json(prompt, WikiDirectionOverview)
        overview = result.overview.strip()
    except Exception as e:
        print(
            f"[wiki] LLM failed on overview of direction "
            f"{direction.name!r}: {e}; leaving overview empty"
        )
        overview = ""

    content = _build_direction_page_content(direction, overview, entities, documents)

    embed_text = "\n\n".join(s for s in (direction.name, overview[:4000]) if s)
    embedding = inputs.embedder.embed(embed_text)

    entity_ids = [e.id for e in entities]
    document_ids = [d.id for d in documents]
    related_page_ids = _resolve_related_page_ids(
        inputs.conn, direction.key, entity_ids, document_ids
    )
    relation_ids = [r.id for r in relations]

    existing = find_wiki_page_for_direction(inputs.conn, direction.key)
    if existing is None:
        insert_wiki_page(
            inputs.conn,
            direction_key=direction.key,
            type="direction",
            entity_id=None,
            document_id=None,
            title=direction.name,
            content=content,
            related_page_ids=related_page_ids,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            embedding=embedding,
        )
    else:
        update_wiki_page(
            inputs.conn,
            existing.id,
            title=direction.name,
            content=content,
            related_page_ids=related_page_ids,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            embedding=embedding,
        )


def _refresh_related_page_ids(
    inputs: StageInputs,
    direction: Direction,
    entities: list[Entity],
    documents: list[Document],
) -> None:
    by_id = {e.id: e for e in entities}
    docs_by_id = {d.id: d for d in documents}

    for entity in entities:
        page = find_wiki_page_for_entity(inputs.conn, direction.key, entity.id)
        if page is None:
            continue
        relations = list_entity_relations_by_entity(inputs.conn, entity.id)
        neighbor_ids: list[int] = []
        doc_ids: list[int] = []
        for rel in relations:
            other = (
                rel.target_entity_id
                if rel.source_entity_id == entity.id
                else rel.source_entity_id
            )
            if other in by_id and other not in neighbor_ids:
                neighbor_ids.append(other)
            if rel.document_id is not None and rel.document_id not in doc_ids:
                if rel.document_id in docs_by_id:
                    doc_ids.append(rel.document_id)
        related = _resolve_related_page_ids(
            inputs.conn, direction.key, neighbor_ids, doc_ids
        )
        update_wiki_page_related_ids(inputs.conn, page.id, related)

    for d in documents:
        page = find_wiki_page_for_document(inputs.conn, direction.key, d.id)
        if page is None:
            continue
        relations = list_entity_relations_by_document(inputs.conn, d.id)
        entity_ids: list[int] = []
        for rel in relations:
            for eid in (rel.source_entity_id, rel.target_entity_id):
                if eid in by_id and eid not in entity_ids:
                    entity_ids.append(eid)
        related = _resolve_related_page_ids(
            inputs.conn, direction.key, entity_ids, []
        )
        update_wiki_page_related_ids(inputs.conn, page.id, related)

    direction_page = find_wiki_page_for_direction(inputs.conn, direction.key)
    if direction_page is not None:
        related = _resolve_related_page_ids(
            inputs.conn,
            direction.key,
            [e.id for e in entities],
            [d.id for d in documents],
        )
        update_wiki_page_related_ids(inputs.conn, direction_page.id, related)


def update_wiki(inputs: StageInputs) -> None:
    if inputs.embedder is None:
        raise ValueError("wiki stage requires an embedder")

    doc = get_document(inputs.conn, inputs.document_id)
    direction = get_direction(inputs.conn, doc.direction_key)

    direction_entities = list_entities_by_direction(inputs.conn, doc.direction_key)
    direction_documents = list_documents_by_direction(inputs.conn, doc.direction_key)
    by_id: dict[int, Entity] = {e.id: e for e in direction_entities}
    docs_by_id: dict[int, Document] = {d.id: d for d in direction_documents}

    abbrev_items = [
        (e.name, e.description) for e in direction_entities if e.is_abbreviation
    ]
    abbrev_block = abbreviation_block.format_for_prompt(abbrev_items)

    affected_entity_ids: list[int] = []
    seen_affected: set[int] = set()
    for stage in list_stage_entities_full(inputs.conn):
        ent = get_entity_by_name(
            inputs.conn, doc.direction_key, stage.type, stage.name
        )
        if ent is not None and ent.id not in seen_affected:
            affected_entity_ids.append(ent.id)
            seen_affected.add(ent.id)
    for rel in list_entity_relations_by_document(inputs.conn, doc.id):
        for eid in (rel.source_entity_id, rel.target_entity_id):
            if eid not in seen_affected:
                affected_entity_ids.append(eid)
                seen_affected.add(eid)

    for entity_id in affected_entity_ids:
        entity = by_id.get(entity_id)
        if entity is None:
            continue
        try:
            _upsert_entity_page(
                inputs,
                doc.direction_key,
                entity,
                by_id,
                docs_by_id,
                abbrev_block,
            )
        except Exception as e:
            print(
                f"[wiki] failed to build wiki page for entity "
                f"{entity.name!r}: {e}; skipping"
            )

    _upsert_document_page(inputs, doc.direction_key, doc, by_id)

    direction_relations = list_entity_relations_by_direction(
        inputs.conn, doc.direction_key
    )
    _upsert_direction_page(
        inputs,
        direction,
        direction_entities,
        direction_documents,
        direction_relations,
        by_id,
        abbrev_block,
    )

    _refresh_related_page_ids(
        inputs, direction, direction_entities, direction_documents
    )

    inputs.conn.commit()
