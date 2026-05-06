CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS llm_wiki_rag;

CREATE TABLE IF NOT EXISTS llm_wiki_rag.directions (
    key          TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    settings     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_wiki_rag.documents (
    id             BIGSERIAL PRIMARY KEY,
    direction_key  TEXT NOT NULL REFERENCES llm_wiki_rag.directions(key),
    title          TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'new',
    content        TEXT NOT NULL,
    use_chunking   BOOLEAN NOT NULL DEFAULT false,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_wiki_rag.document_chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES llm_wiki_rag.documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
