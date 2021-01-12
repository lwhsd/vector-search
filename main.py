from typing import Optional
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tfidf import get_paragraph
from faiss_indobert import search_paragraph
import os
from utils.load_pdf import get_text_from_pdf
from utils.generate_faiss_index_indobert import generate_faissbert_pickle

SAVED_FILES = 'indobert_files'
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

@app.post("/train-indobert/")
async def create_upload_file(file: UploadFile = File(...)):

    if not os.path.isdir(SAVED_FILES):
        os.mkdir(SAVED_FILES)

    file_location = os.path.join(SAVED_FILES, file.filename)
    # if (not str(os.path.exists(file_location))):
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    _, csv_file = get_text_from_pdf(file_location)
    pickle_file = generate_faissbert_pickle(csv_file)

    return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    # return {"filename": file.filename}