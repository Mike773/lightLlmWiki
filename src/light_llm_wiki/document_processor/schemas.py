from pydantic import BaseModel


class AbbreviationsList(BaseModel):
    items: list[str]


class ExpansionLookup(BaseModel):
    expansion: str | None
