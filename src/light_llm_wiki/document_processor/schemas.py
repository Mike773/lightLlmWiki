from pydantic import BaseModel


class AbbreviationItem(BaseModel):
    name: str
    expansion: str | None


class AbbreviationsExtraction(BaseModel):
    items: list[AbbreviationItem]
