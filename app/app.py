from fastapi import FastAPI
from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str
    context: str | None = None


class SearchQueryResponse(BaseModel):
    query: str
    result: str
    context: str | None = None


app = FastAPI()


@app.post("/query/")
def read_root(search_query: SearchQuery) -> SearchQueryResponse:
    return SearchQueryResponse(
        query=search_query.query,
        context=search_query.context,
        result=""
    )
