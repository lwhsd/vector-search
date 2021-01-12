from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from tfidf import get_paragraph
from faiss_indobert import search_paragraph

class Text(BaseModel):
    query : str
    n: Optional[int] = None

app = FastAPI()

@app.get('/')
def docs_redirect():
    return RedirectResponse('/docs')

@app.post("/tf-idf/")
async def create_item(query: Text):
    par = get_paragraph(query.query, query.n)
    result =  par.reset_index().to_dict(orient='records')
    return result

@app.post("/faiss-indobert/")
async def create_item(query: Text):
    par = search_paragraph(query.query, query.n)
    result =  par.reset_index().to_dict(orient='records')
    return result