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
