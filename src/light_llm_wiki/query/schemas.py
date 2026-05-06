from pydantic import BaseModel


class QueryAnswer(BaseModel):
    answer: str
    unsupported: list[str]
