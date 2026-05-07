from pydantic import BaseModel, Field


class QueryAnswer(BaseModel):
    answer: str
    unsupported: list[str] = Field(default_factory=list)


class QueryStory(BaseModel):
    story: str


class DocumentRelevance(BaseModel):
    relevant: bool
    summary: str = ""
