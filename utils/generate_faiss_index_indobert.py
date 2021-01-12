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
import timeit

os.environ['KMP_DUPLICATE_LIB_OK']='True'

MAX_TOKEN_LENTH=59
FILE_PATH = "laporan-keuangan-2018-paragraph.csv"
SAVED_FILES = 'indobert_files'

# def vector_search(query, model, index, num_results=10):
#     vector = model.encode(list(query))
#     D, I = index.search(np.array(vector).astype("float32"), k=num_results)
#     return D, I

# def vector_search_indo(query, model, tokenizer, index, num_results=10):
#     query=tokenizer.encode(query,truncation=True,max_length=512, add_special_tokens=True)
#     tokens = [query]
#     max_len = 0
#     for i in tokens:
#         if len(i) > max_len:
#             max_len = len(i)

#     padded = np.array([i + [0]*(max_len-len(i)) for i in tokens])

#     attention_mask = np.where(padded != 0, 1, 0)

#     input_ids = torch.tensor(padded)
#     attention_mask = torch.tensor(attention_mask)

#     with torch.no_grad():
#         last_hidden_states = model(input_ids, attention_mask=attention_mask)

#     features = last_hidden_states[0][:,0,:].numpy()
#     D, I = index.search(np.array(features).astype("float32"), k=num_results)
#     return D, I


# def id2details(df, I, column):
#     """Returns the paper titles based on the paper index."""
#     return [list(df[df.id == idx][column]) for idx in I[0]]

def generate_faissbert_pickle(file_path):
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    model = AutoModel.from_pretrained("indolem/indobert-base-uncased")

    paragraph = pd.read_csv(file_path)
    tokens = []
    for i, row in paragraph.iterrows():
        tokendata =tokenizer.encode(row['Content'],truncation=True, max_length=512,add_special_tokens=True)
        tokens.append(tokendata)

    max_len = 0
    for i in tokens:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokens])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    start = timeit.default_timer()
    print('Generate features bert start....................')
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()
    stop = timeit.default_timer()
    print('Generate features bert finished in :: ', stop - start)


    print ("************faiss start**************")
    embeddings = np.array([embedding for embedding in features]).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])

    # Step 3: Pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)

    # Step 4: Add vectors and their IDs
    index.add_with_ids(embeddings, paragraph.id.values)

    print(f"Number of vectors in the Faiss index: {index.ntotal}")

    print(paragraph.iloc[2, 2])

    D, I = index.search(np.array([embeddings[2]]), k=3)
    print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')

    f = open(file_path)
    fine_name = os.path.basename(f.name).split('.')
   
    with open(SAVED_FILES+'/'+fine_name[0]+".pickle", "wb") as h:
        pickle.dump(faiss.serialize_index(index), h)
    
    return SAVED_FILES+'/'+fine_name[0]+".pickle"
