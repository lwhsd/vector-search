from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from scipy import spatial
import os
import faiss
import numpy as np
import pickle
from pathlib import Path
import pandas as pd

# pd.set_option('display.max_colwidth', None)
CSV_PATH = "faiss_indobert_docs/laporan-keuangan-2018-paragraph.csv"
N_RECOMMENDED = 10
PATH_TO_FAISS_PICKLE = "faiss_indobert_docs/faiss_index.pickle"

def id2details(df, I, column):
    return [list(df[df.id == idx][column]) for idx in I[0]]

def id2details_df(df, I):
    mask = df['id'].isin(I[0])
    return df.loc[mask]

def vector_search_indo(query, model, tokenizer, index, num_results=5):
    query=tokenizer.encode(query,truncation=True,max_length=512, add_special_tokens=True)
    tokens = [query]
    max_len = 0
    for i in tokens:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokens])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()
    D, I = index.search(np.array(features).astype("float32"), k=num_results)
    return D, I

def read_data(data="faiss_indobert_docs/laporan-keuangan-2018-paragraph.csv"):
    return pd.read_csv(data)

def load_indobert_model():
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
    return model, tokenizer

def load_faiss_index(path_to_faiss="faiss_indobert_docs/faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)

def search_paragraph(query, n=N_RECOMMENDED):
    data = read_data(CSV_PATH)
    model, tokenizer = load_indobert_model()
    faiss_index = load_faiss_index(PATH_TO_FAISS_PICKLE)

    D, I = vector_search_indo(query, model, tokenizer, faiss_index, n)

    print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')
    # res = id2details(data, I, 'Content')
    res = id2details_df(data, I)
    return res


# print (search_paragraph("akuntabilitas keuangan negara", 1))
