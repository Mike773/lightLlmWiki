from dataclasses import dataclass
from datetime import datetime

import psycopg


def get_connection(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)


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


def insert_stage_entity(
    conn: psycopg.Connection,
    *,
    type: str,
    name: str,
    description: str | None,
    related_entity_ids: list[int],
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_wiki_rag.stage_entities
                (type, name, description, related_entity_ids)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (type, name, description, related_entity_ids),
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
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE llm_wiki_rag.stage_entities
            SET description = %s,
                related_entity_ids = %s
            WHERE id = %s
            """,
            (description, related_entity_ids, stage_id),
        )
