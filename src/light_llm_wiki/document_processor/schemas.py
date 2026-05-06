from pydantic import BaseModel, field_validator


class AbbreviationsList(BaseModel):
    items: list[str]


class ExpansionLookup(BaseModel):
    expansion: str | None

    @field_validator("expansion", mode="before")
    @classmethod
    def _normalize_missing(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value


class EntitiesList(BaseModel):
    items: list[str]


class EntityDescription(BaseModel):
    description: str | None

    @field_validator("description", mode="before")
    @classmethod
    def _normalize_missing(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value
