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

CREATE TABLE IF NOT EXISTS llm_wiki_rag.entities (
    id               BIGSERIAL PRIMARY KEY,
    direction_key    TEXT NOT NULL REFERENCES llm_wiki_rag.directions(key),
    type             TEXT NOT NULL,
    name             TEXT NOT NULL,
    description      TEXT,
    is_abbreviation  BOOLEAN NOT NULL DEFAULT false,
    has_expansion    BOOLEAN NOT NULL DEFAULT false,
    embedding        vector(2560),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_wiki_rag.entity_documents (
    entity_id    BIGINT NOT NULL REFERENCES llm_wiki_rag.entities(id) ON DELETE CASCADE,
    document_id  BIGINT NOT NULL REFERENCES llm_wiki_rag.documents(id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id, document_id)
);

CREATE TABLE IF NOT EXISTS llm_wiki_rag.entity_relations (
    id                BIGSERIAL PRIMARY KEY,
    direction_key     TEXT NOT NULL REFERENCES llm_wiki_rag.directions(key),
    source_entity_id  BIGINT NOT NULL REFERENCES llm_wiki_rag.entities(id) ON DELETE CASCADE,
    target_entity_id  BIGINT NOT NULL REFERENCES llm_wiki_rag.entities(id) ON DELETE CASCADE,
    relation_type     TEXT NOT NULL,
    description       TEXT,
    document_id       BIGINT REFERENCES llm_wiki_rag.documents(id) ON DELETE SET NULL,
    quote             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_wiki_rag.wiki_pages (
    id                BIGSERIAL PRIMARY KEY,
    direction_key     TEXT NOT NULL REFERENCES llm_wiki_rag.directions(key),
    type              TEXT NOT NULL,
    entity_id         BIGINT REFERENCES llm_wiki_rag.entities(id) ON DELETE SET NULL,
    document_id       BIGINT REFERENCES llm_wiki_rag.documents(id) ON DELETE SET NULL,
    title             TEXT NOT NULL,
    content           TEXT NOT NULL,
    related_page_ids  BIGINT[] NOT NULL DEFAULT '{}',
    entity_ids        BIGINT[] NOT NULL DEFAULT '{}',
    relation_ids      BIGINT[] NOT NULL DEFAULT '{}',
    embedding         vector(2560),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_wiki_rag.stage_entities (
    id                     BIGSERIAL PRIMARY KEY,
    type                   TEXT NOT NULL,
    name                   TEXT NOT NULL,
    description            TEXT,
    related_entity_ids     BIGINT[] NOT NULL DEFAULT '{}',
    related_wiki_page_ids  BIGINT[] NOT NULL DEFAULT '{}',
    embedding              vector(2560),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_wiki_rag.stage_relations (
    id                BIGSERIAL PRIMARY KEY,
    source_stage_id   BIGINT NOT NULL REFERENCES llm_wiki_rag.stage_entities(id) ON DELETE CASCADE,
    target_stage_id   BIGINT NOT NULL REFERENCES llm_wiki_rag.stage_entities(id) ON DELETE CASCADE,
    relation_type     TEXT NOT NULL,
    description       TEXT,
    quote             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE llm_wiki_rag.entities
    ADD COLUMN IF NOT EXISTS embedding vector(2560);

ALTER TABLE llm_wiki_rag.stage_entities
    ADD COLUMN IF NOT EXISTS embedding vector(2560);

ALTER TABLE llm_wiki_rag.stage_relations
    ADD COLUMN IF NOT EXISTS quote TEXT;

ALTER TABLE llm_wiki_rag.wiki_pages
    ADD COLUMN IF NOT EXISTS type TEXT;

ALTER TABLE llm_wiki_rag.wiki_pages
    ADD COLUMN IF NOT EXISTS entity_id BIGINT REFERENCES llm_wiki_rag.entities(id) ON DELETE SET NULL;

ALTER TABLE llm_wiki_rag.wiki_pages
    ADD COLUMN IF NOT EXISTS document_id BIGINT REFERENCES llm_wiki_rag.documents(id) ON DELETE SET NULL;

ALTER TABLE llm_wiki_rag.wiki_pages
    ADD COLUMN IF NOT EXISTS embedding vector(2560);
