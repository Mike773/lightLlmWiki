from dataclasses import dataclass, field

import psycopg

from light_llm_wiki.db import (
    Document,
    Entity,
    EntityRelation,
    find_nearest_entity_by_type,
    get_entity_by_name_ci,
    list_documents_by_ids,
    list_entity_relations_by_entity,
)
from light_llm_wiki.document_processor.schemas import (
    AbbreviationsList,
    EntitiesList,
)
from light_llm_wiki.embedding import EmbeddingClient
from light_llm_wiki.llm import LLMClient
from light_llm_wiki.query.prompts import (
    ANSWER_PROMPT,
    NARRATIVE_PROMPT,
    QUESTION_ABBREVIATIONS_PROMPT,
    QUESTION_ENTITIES_PROMPT,
)
from light_llm_wiki.query.schemas import QueryAnswer, QueryStory


@dataclass
class ResolvedItem:
    requested_name: str
    entity: Entity
    via: str  # 'exact' | 'embedding'


@dataclass
class QueryResult:
    question: str
    abbreviations_found: list[ResolvedItem] = field(default_factory=list)
    abbreviations_missing: list[str] = field(default_factory=list)
    entities_found: list[ResolvedItem] = field(default_factory=list)
    entities_missing: list[str] = field(default_factory=list)
    relations: list[EntityRelation] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    answer: str = ""
    unsupported: list[str] = field(default_factory=list)
    trace: str = ""
    story: str | None = None


def _resolve_one(
    conn: psycopg.Connection,
    embedder: EmbeddingClient,
    direction_key: str,
    type_: str,
    name: str,
    embedding_threshold: float,
) -> ResolvedItem | None:
    exact = get_entity_by_name_ci(conn, direction_key, type_, name)
    if exact is not None:
        return ResolvedItem(requested_name=name, entity=exact, via="exact")

    vec = embedder.embed(name)
    nearest = find_nearest_entity_by_type(
        conn, direction_key, type_, vec, embedding_threshold
    )
    if nearest is not None:
        return ResolvedItem(requested_name=name, entity=nearest, via="embedding")

    return None


def _format_abbreviations_to_exclude(items: list[ResolvedItem], missing: list[str]) -> str:
    names = [item.requested_name for item in items] + missing
    if not names:
        return "(в вопросе аббревиатур не найдено)"
    return "\n".join(f"- {n}" for n in names)


def _format_abbreviations_block(items: list[ResolvedItem]) -> str:
    if not items:
        return "(аббревиатур не нашлось)"
    lines = []
    for item in items:
        ent = item.entity
        desc = ent.description or "(расшифровка не зафиксирована)"
        lines.append(f"- {ent.name} — {desc}")
    return "\n".join(lines)


def _format_entities_block(
    primary: list[ResolvedItem],
    neighbors: list[Entity],
) -> str:
    seen: set[int] = set()
    lines: list[str] = []
    for item in primary:
        ent = item.entity
        if ent.id in seen:
            continue
        seen.add(ent.id)
        desc = ent.description or "(описание не зафиксировано)"
        lines.append(f"- {ent.name} ({ent.type}) — {desc}")
    for ent in neighbors:
        if ent.id in seen:
            continue
        seen.add(ent.id)
        desc = ent.description or "(описание не зафиксировано)"
        lines.append(f"- {ent.name} ({ent.type}) — {desc}")
    return "\n".join(lines) if lines else "(сущностей не нашлось)"


def _format_relations_block(
    relations: list[EntityRelation], by_id: dict[int, Entity]
) -> str:
    if not relations:
        return "(связей не нашлось)"
    lines = []
    for rel in relations:
        src = by_id.get(rel.source_entity_id)
        tgt = by_id.get(rel.target_entity_id)
        if src is None or tgt is None:
            continue
        line = f"- {src.name} → {tgt.name} — {rel.relation_type}"
        if rel.description:
            line += f": {rel.description}"
        if rel.quote:
            line += f"\n  Цитата: «{rel.quote}»"
        lines.append(line)
    return "\n".join(lines) if lines else "(связей не нашлось)"


def _format_documents_block(documents: list[Document]) -> str:
    if not documents:
        return "(документов не нашлось)"
    parts = []
    for d in documents:
        parts.append(f"### {d.title}\n{d.content}")
    return "\n\n".join(parts)


def _format_missing_block(items: list[str]) -> str:
    if not items:
        return "(нет)"
    return "\n".join(f"- {n}" for n in items)


def _format_unsupported_block(items: list[str]) -> str:
    if not items:
        return "(нет — контекст покрывает вопрос полностью)"
    return "\n".join(f"- {n}" for n in items)


def _render_trace(result: QueryResult, by_id: dict[int, Entity]) -> str:
    lines: list[str] = []
    lines.append("## Анализ вопроса")
    lines.append("")

    abbr_names = [i.requested_name for i in result.abbreviations_found] + result.abbreviations_missing
    if not abbr_names:
        lines.append("В вопросе аббревиатур я не выделил.")
    else:
        lines.append(f"Из вопроса я выделил аббревиатуры: {', '.join(abbr_names)}.")
        for item in result.abbreviations_found:
            ent = item.entity
            via = "точно" if item.via == "exact" else "по смыслу (embedding)"
            desc = ent.description or "расшифровка не зафиксирована"
            lines.append(
                f"- «{item.requested_name}» — нашёл {via}: {ent.name}. "
                f"Расшифровка: {desc}."
            )
        for name in result.abbreviations_missing:
            lines.append(f"- «{name}» — в базе не нашёл.")

    lines.append("")
    ent_names = [i.requested_name for i in result.entities_found] + result.entities_missing
    if not ent_names:
        lines.append("В вопросе сущностей я не выделил.")
    else:
        lines.append(f"Сущности из вопроса: {', '.join(ent_names)}.")
        for item in result.entities_found:
            ent = item.entity
            via = "точно" if item.via == "exact" else "по смыслу (embedding)"
            desc = ent.description or "описание не зафиксировано"
            lines.append(
                f"- «{item.requested_name}» — нашёл {via}: {ent.name}. {desc}"
            )
        for name in result.entities_missing:
            lines.append(f"- «{name}» — в базе не нашёл.")

    primary_ids = {i.entity.id for i in result.abbreviations_found} | {
        i.entity.id for i in result.entities_found
    }

    if result.relations:
        lines.append("")
        lines.append("## Что нашлось по этим сущностям")
        lines.append("")
        relations_by_primary: dict[int, list[EntityRelation]] = {}
        for rel in result.relations:
            for eid in (rel.source_entity_id, rel.target_entity_id):
                if eid in primary_ids:
                    relations_by_primary.setdefault(eid, []).append(rel)
        for eid, rels in relations_by_primary.items():
            ent = by_id.get(eid)
            if ent is None:
                continue
            lines.append(f"У сущности «{ent.name}» нашёл связи:")
            for rel in rels:
                other_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == eid
                    else rel.source_entity_id
                )
                other = by_id.get(other_id)
                if other is None:
                    continue
                line = f"- с «{other.name}» (тип: {rel.relation_type})"
                if rel.description:
                    line += f". {rel.description}"
                if rel.quote:
                    line += f"\n  Цитата: «{rel.quote}»"
                lines.append(line)
            lines.append("")

    if result.documents:
        lines.append("## Документы, которые могут содержать ответ")
        lines.append("")
        for d in result.documents:
            mention_names: list[str] = []
            for rel in result.relations:
                if rel.document_id != d.id:
                    continue
                for eid in (rel.source_entity_id, rel.target_entity_id):
                    if eid in primary_ids:
                        ent = by_id.get(eid)
                        if ent is not None and ent.name not in mention_names:
                            mention_names.append(ent.name)
            if mention_names:
                lines.append(
                    f"- «{d.title}» — упоминает: {', '.join(mention_names)}."
                )
            else:
                lines.append(f"- «{d.title}»")
        lines.append("")

        lines.append("## Что нашлось в этих документах")
        lines.append("")
        for d in result.documents:
            quotes: list[str] = []
            for rel in result.relations:
                if rel.document_id == d.id and rel.quote and rel.quote not in quotes:
                    quotes.append(rel.quote)
            if quotes:
                lines.append(f"В документе «{d.title}»:")
                for q in quotes:
                    lines.append(f"> {q}")
                lines.append("")

    if not result.abbreviations_found and not result.entities_found:
        lines.append("")
        lines.append(
            "По выделенным понятиям в базе знаний ничего не нашлось."
        )

    lines.append("## Ответ")
    lines.append("")
    lines.append(result.answer)

    if result.unsupported:
        lines.append("")
        lines.append("Часть вопроса не подтверждается базой:")
        for item in result.unsupported:
            lines.append(f"- {item}")

    return "\n".join(lines).rstrip() + "\n"


def answer_question(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: EmbeddingClient,
    direction_key: str,
    question: str,
    *,
    embedding_threshold: float = 0.5,
    narrate: bool = False,
) -> QueryResult:
    result = QueryResult(question=question)

    try:
        abbr_extraction = llm.complete_json(
            QUESTION_ABBREVIATIONS_PROMPT.format(question=question),
            AbbreviationsList,
        )
    except Exception as e:
        print(f"[query] LLM failed on extracting abbreviations: {e}; assuming none")
        abbr_extraction = AbbreviationsList()

    for name in abbr_extraction.items:
        item = _resolve_one(
            conn, embedder, direction_key, "abbreviation", name, embedding_threshold
        )
        if item is None:
            result.abbreviations_missing.append(name)
        else:
            result.abbreviations_found.append(item)

    abbr_block_for_entities = _format_abbreviations_to_exclude(
        result.abbreviations_found, result.abbreviations_missing
    )
    try:
        ent_extraction = llm.complete_json(
            QUESTION_ENTITIES_PROMPT.format(
                question=question,
                abbreviations_to_exclude=abbr_block_for_entities,
            ),
            EntitiesList,
        )
    except Exception as e:
        print(f"[query] LLM failed on extracting entities: {e}; assuming none")
        ent_extraction = EntitiesList()

    for name in ent_extraction.items:
        item = _resolve_one(
            conn, embedder, direction_key, "entity", name, embedding_threshold
        )
        if item is None:
            result.entities_missing.append(name)
        else:
            result.entities_found.append(item)

    primary_entities: list[Entity] = []
    seen_primary: set[int] = set()
    for item in result.abbreviations_found + result.entities_found:
        if item.entity.id in seen_primary:
            continue
        seen_primary.add(item.entity.id)
        primary_entities.append(item.entity)

    relations: list[EntityRelation] = []
    seen_rel_ids: set[int] = set()
    neighbor_ids: set[int] = set()
    document_ids: set[int] = set()
    for ent in primary_entities:
        for rel in list_entity_relations_by_entity(conn, ent.id):
            if rel.id in seen_rel_ids:
                continue
            seen_rel_ids.add(rel.id)
            relations.append(rel)
            other = (
                rel.target_entity_id
                if rel.source_entity_id == ent.id
                else rel.source_entity_id
            )
            if other not in seen_primary:
                neighbor_ids.add(other)
            if rel.document_id is not None:
                document_ids.add(rel.document_id)

    neighbors: list[Entity] = []
    if neighbor_ids:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, direction_key, type, name, description,
                       is_abbreviation, has_expansion, created_at
                FROM llm_wiki_rag.entities
                WHERE id = ANY(%s)
                ORDER BY id
                """,
                (list(neighbor_ids),),
            )
            for row in cur.fetchall():
                neighbors.append(Entity(*row))

    documents = list_documents_by_ids(conn, sorted(document_ids))

    by_id: dict[int, Entity] = {e.id: e for e in primary_entities + neighbors}

    abbreviations_block = _format_abbreviations_block(result.abbreviations_found)
    entities_block = _format_entities_block(result.entities_found, neighbors)
    relations_block = _format_relations_block(relations, by_id)
    documents_block = _format_documents_block(documents)

    try:
        answer_obj = llm.complete_json(
            ANSWER_PROMPT.format(
                question=question,
                abbreviations_block=abbreviations_block,
                entities_block=entities_block,
                relations_block=relations_block,
                documents_block=documents_block,
            ),
            QueryAnswer,
        )
        answer_text = answer_obj.answer
        unsupported = answer_obj.unsupported
    except Exception as e:
        print(f"[query] LLM failed on the final answer: {e}")
        answer_text = (
            "Не удалось получить ответ от модели — возможно, контекст слишком "
            "велик или модель вернула невалидный JSON. Сырые материалы по "
            "вопросу собраны выше в трассе."
        )
        unsupported = []

    result.relations = relations
    result.documents = documents
    result.answer = answer_text
    result.unsupported = unsupported
    result.trace = _render_trace(result, by_id)

    if narrate:
        narrative_prompt = NARRATIVE_PROMPT.format(
            question=question,
            abbreviations_found_block=_format_abbreviations_block(
                result.abbreviations_found
            ),
            abbreviations_missing_block=_format_missing_block(
                result.abbreviations_missing
            ),
            entities_found_block=_format_entities_block(
                result.entities_found, []
            ),
            entities_missing_block=_format_missing_block(result.entities_missing),
            relations_block=relations_block,
            documents_block=documents_block,
            answer=answer_text,
            unsupported_block=_format_unsupported_block(unsupported),
        )
        try:
            story_obj = llm.complete_json(narrative_prompt, QueryStory)
            result.story = story_obj.story.strip()
        except Exception as e:
            print(f"[query] LLM failed on the narrative: {e}")
            result.story = (
                "Не удалось собрать связный рассказ — модель вернула невалидный "
                "ответ. Структурированная трасса доступна в поле trace."
            )

    return result
