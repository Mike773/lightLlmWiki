from dataclasses import dataclass
from datetime import datetime

import psycopg
from pgvector.psycopg import register_vector


def get_connection(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn


@dataclass
class Document:
    id: int
    direction_key: str
    title: str
    status: str
    content: str
    use_chunking: bool
    created_at: datetime


@dataclass
class Entity:
    id: int
    direction_key: str
    type: str
    name: str
    description: str | None
    is_abbreviation: bool
    has_expansion: bool
    created_at: datetime


@dataclass
class StageEntity:
    id: int
    type: str
    name: str
    description: str | None


@dataclass
class StageRelation:
    id: int
    source_stage_id: int
    target_stage_id: int
    relation_type: str
    description: str | None
    quote: str | None


@dataclass
class StageEntityFull:
    id: int
    type: str
    name: str
    description: str | None
    embedding: list[float] | None


@dataclass
class EntityRelation:
    id: int
    direction_key: str
    source_entity_id: int
    target_entity_id: int
    relation_type: str
    description: str | None
    document_id: int | None
    quote: str | None
    created_at: datetime


def get_document(conn: psycopg.Connection, document_id: int) -> Document:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, direction_key, title, status, content, use_chunking, created_at
            FROM llm_wiki_rag.documents
            WHERE id = %s
            """,
            (document_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"document {document_id} not found")
    return Document(*row)


def find_abbreviation_entities_by_name(
    conn: psycopg.Connection, direction_key: str, name: str
) -> list[Entity]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, direction_key, type, name, description,
                   is_abbreviation, has_expansion, created_at
            FROM llm_wiki_rag.entities
            WHERE direction_key = %s
              AND name = %s
              AND is_abbreviation = true
            """,
            (direction_key, name),
        )
        rows = cur.fetchall()
    return [Entity(*row) for row in rows]


def clear_stage(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM llm_wiki_rag.stage_relations")
        cur.execute("DELETE FROM llm_wiki_rag.stage_entities")


def clear_stage_by_type(conn: psycopg.Connection, type_: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM llm_wiki_rag.stage_entities WHERE type = %s",
            (type_,),
        )


def list_stage_abbreviations(
    conn: psycopg.Connection,
) -> list[tuple[str, str | None]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT name, description
            FROM llm_wiki_rag.stage_entities
            WHERE type = 'abbreviation'
            ORDER BY id
            """
        )
        rows = cur.fetchall()
    return [(name, description) for name, description in rows]


def insert_stage_entity(
    conn: psycopg.Connection,
    *,
    type: str,
    name: str,
    description: str | None,
    related_entity_ids: list[int],
    embedding: list[float] | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_wiki_rag.stage_entities
                (type, name, description, related_entity_ids, embedding)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (type, name, description, related_entity_ids, embedding),
        )
        row = cur.fetchone()
    assert row is not None
    return row[0]


def update_stage_entity(
    conn: psycopg.Connection,
    stage_id: int,
    *,
    description: str | None,
    related_entity_ids: list[int],
    embedding: list[float] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE llm_wiki_rag.stage_entities
            SET description = %s,
                related_entity_ids = %s,
                embedding = COALESCE(%s::vector, embedding)
            WHERE id = %s
            """,
            (description, related_entity_ids, embedding, stage_id),
        )


def find_similar_entities(
    conn: psycopg.Connection,
    direction_key: str,
    embedding: list[float],
    limit: int = 10,
) -> list[Entity]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, direction_key, type, name, description,
                   is_abbreviation, has_expansion, created_at
            FROM llm_wiki_rag.entities
            WHERE direction_key = %s
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (direction_key, embedding, limit),
        )
        rows = cur.fetchall()
    return [Entity(*row) for row in rows]


def list_stage_entities(conn: psycopg.Connection) -> list[StageEntity]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, type, name, description
            FROM llm_wiki_rag.stage_entities
            ORDER BY id
            """
        )
        rows = cur.fetchall()
    return [StageEntity(*row) for row in rows]


def clear_stage_relations(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM llm_wiki_rag.stage_relations")


def insert_stage_relation(
    conn: psycopg.Connection,
    *,
    source_stage_id: int,
    target_stage_id: int,
    relation_type: str,
    description: str | None = None,
    quote: str | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_wiki_rag.stage_relations
                (source_stage_id, target_stage_id, relation_type, description, quote)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (source_stage_id, target_stage_id, relation_type, description, quote),
        )
        row = cur.fetchone()
    assert row is not None
    return row[0]


def update_stage_relation(
    conn: psycopg.Connection,
    stage_id: int,
    *,
    description: str | None,
    quote: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE llm_wiki_rag.stage_relations
            SET description = %s,
                quote = %s
            WHERE id = %s
            """,
            (description, quote, stage_id),
        )


def delete_stage_relation(conn: psycopg.Connection, stage_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM llm_wiki_rag.stage_relations WHERE id = %s",
            (stage_id,),
        )


def list_stage_entities_full(conn: psycopg.Connection) -> list[StageEntityFull]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, type, name, description, embedding
            FROM llm_wiki_rag.stage_entities
            ORDER BY id
            """
        )
        rows = cur.fetchall()
    return [StageEntityFull(*row) for row in rows]


def list_stage_relations(conn: psycopg.Connection) -> list[StageRelation]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_stage_id, target_stage_id, relation_type,
                   description, quote
            FROM llm_wiki_rag.stage_relations
            ORDER BY id
            """
        )
        rows = cur.fetchall()
    return [StageRelation(*row) for row in rows]


def get_entity_by_name(
    conn: psycopg.Connection, direction_key: str, type_: str, name: str
) -> Entity | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, direction_key, type, name, description,
                   is_abbreviation, has_expansion, created_at
            FROM llm_wiki_rag.entities
            WHERE direction_key = %s AND type = %s AND name = %s
            """,
            (direction_key, type_, name),
        )
        row = cur.fetchone()
    return Entity(*row) if row else None


def insert_entity(
    conn: psycopg.Connection,
    *,
    direction_key: str,
    type: str,
    name: str,
    description: str | None,
    is_abbreviation: bool,
    has_expansion: bool,
    embedding: list[float] | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_wiki_rag.entities
                (direction_key, type, name, description,
                 is_abbreviation, has_expansion, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                direction_key, type, name, description,
                is_abbreviation, has_expansion, embedding,
            ),
        )
        row = cur.fetchone()
    assert row is not None
    return row[0]


def update_entity(
    conn: psycopg.Connection,
    entity_id: int,
    *,
    description: str | None,
    is_abbreviation: bool,
    has_expansion: bool,
    embedding: list[float] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE llm_wiki_rag.entities
            SET description = %s,
                is_abbreviation = %s,
                has_expansion = %s,
                embedding = COALESCE(%s::vector, embedding)
            WHERE id = %s
            """,
            (description, is_abbreviation, has_expansion, embedding, entity_id),
        )


def find_entity_relation_by_pair(
    conn: psycopg.Connection,
    direction_key: str,
    a_entity_id: int,
    b_entity_id: int,
) -> EntityRelation | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, direction_key, source_entity_id, target_entity_id,
                   relation_type, description, document_id, quote, created_at
            FROM llm_wiki_rag.entity_relations
            WHERE direction_key = %s
              AND ((source_entity_id = %s AND target_entity_id = %s)
                OR (source_entity_id = %s AND target_entity_id = %s))
            ORDER BY id
            LIMIT 1
            """,
            (direction_key, a_entity_id, b_entity_id, b_entity_id, a_entity_id),
        )
        row = cur.fetchone()
    return EntityRelation(*row) if row else None


def insert_entity_relation(
    conn: psycopg.Connection,
    *,
    direction_key: str,
    source_entity_id: int,
    target_entity_id: int,
    relation_type: str,
    description: str | None,
    document_id: int | None,
    quote: str | None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_wiki_rag.entity_relations
                (direction_key, source_entity_id, target_entity_id,
                 relation_type, description, document_id, quote)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                direction_key, source_entity_id, target_entity_id,
                relation_type, description, document_id, quote,
            ),
        )
        row = cur.fetchone()
    assert row is not None
    return row[0]


def update_entity_relation(
    conn: psycopg.Connection,
    relation_id: int,
    *,
    description: str | None,
    document_id: int | None,
    quote: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE llm_wiki_rag.entity_relations
            SET description = %s,
                document_id = %s,
                quote = %s
            WHERE id = %s
            """,
            (description, document_id, quote, relation_id),
        )
