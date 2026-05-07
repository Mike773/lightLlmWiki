from pydantic import BaseModel, Field, field_validator


class AbbreviationsList(BaseModel):
    items: list[str] = Field(default_factory=list)


class ExpansionLookup(BaseModel):
    expansion: str | None = None
    quote: str | None = None

    @field_validator("expansion", "quote", mode="before")
    @classmethod
    def _normalize_missing(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value


class EntitiesList(BaseModel):
    items: list[str] = Field(default_factory=list)


class EntityDescription(BaseModel):
    description: str | None = None

    @field_validator("description", mode="before")
    @classmethod
    def _normalize_missing(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value


class RelationCandidate(BaseModel):
    target_id: int
    relation_type: str


class RelationCandidatesList(BaseModel):
    items: list[RelationCandidate] = Field(default_factory=list)


class RelationLookup(BaseModel):
    description: str | None = None
    quote: str | None = None

    @field_validator("description", "quote", mode="before")
    @classmethod
    def _normalize_missing(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value


class WikiEntitySummary(BaseModel):
    summary: str


class WikiDirectionOverview(BaseModel):
    overview: str
