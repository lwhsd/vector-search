#pip install Sastrawi
#pip install wordcloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover,ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np #Operasi Matematika dan linear aljebra
import pandas as pd #data processing
import matplotlib.pyplot as plt #Visualisasi data
import seaborn as sns #Visualisasi data
import string
import nltk
from nltk.tokenize import RegexpTokenizer 
nltk.download('punkt')
from nltk.corpus import stopwords 
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time
from scipy import spatial
import math
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import heapq
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os

# pd.set_option('display.max_colwidth', None)
RAW_FILE_PATH = "/Users/admin/Code/MINE/hackathon/bpk_2021/laporan-keuangan-2018-paragraph.csv"
CLEAN_FILE_PATH = "/Users/admin/Code/MINE/hackathon/bpk_2021/laporan-keuangan-2018-clean.csv"
BASE_PATH = "/Users/admin/Code/MINE/hackathon/bpk_2021/financeReport"

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
default_stopwords = StopWordRemoverFactory().get_stop_words()
additional_stopwords=["(",")","senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
dictionary=ArrayDictionary(default_stopwords+additional_stopwords)
id_stopword = StopWordRemover(dictionary)
en_stopword = set(stopwords.words('english'))
en_stemmer = PorterStemmer()

def remove_numbers(text):
  words=tokenizer.tokenize(text)
  return " ".join(words)

def remove_punctuation(text):
  words = text.split()
  table = str.maketrans('', '', string.punctuation)
  stripped = [w.translate(table) for w in words]
  return " ".join(stripped)

def stem_text(text):
  return stemmer.stem(text)

def remove_stopwords(text):
  return id_stopword.remove(text)

def remove_english_stopwords(text):
  if text:
    return " ".join([token for token in text.split() if token not in en_stopword])

def stem_english_text(text):
  return " ".join([en_stemmer.stem(word) for word in text.split()])

def remove_single_char(text):
  return " ".join([ x for x in text.split() if len(x)>1])

def clean_text(file_path="/Users/admin/Code/MINE/hackathon/bpk_2021/laporan-keuangan-2018-paragraph.csv"):
    task = pd.read_csv(file_path)
    task=task[task.Content.notna()]
    task.Content=task.Content.str.lower()
    task.Content=task.Content.apply(remove_punctuation)
    task.Content=task.Content.apply(remove_stopwords)
    task.Content=task.Content.apply(stem_text)
    task.Content=task.Content.apply(remove_english_stopwords)
    task=task[task.Content.notna()]
    task.Content=task.Content.apply(stem_english_text)
    task.Content=task.Content.apply(remove_single_char)
    task=task[task.Content.notna()]
    task.to_csv('laporan-keuangan-2018-clean.csv')

    print (task.head(10))

def train_tfidfvactorizer(file_path="/Users/admin/Code/MINE/hackathon/bpk_2021/laporan-keuangan-2018-clean.csv"):
    try:
        task = pd.read_csv(file_path)
        content_data = task.Content
        id_label = task.id    

        tfidf_vectorizer=TfidfVectorizer(use_idf=True)
        fitted_tfidf_vectorizer = tfidf_vectorizer.fit(content_data.to_list())

        with open('tfidf_model_path.pickle', "wb") as fo:
            pickle.dump(fitted_tfidf_vectorizer, fo)
            print("SAVED tfidf model")

        paragraph =content_data.to_list()
        tfidf_vectorizer_vectors = fitted_tfidf_vectorizer.transform(paragraph)
        content_vecs=tfidf_vectorizer_vectors.toarray()
        vecs=np.column_stack([content_vecs,id_label])
        dfres = pd.DataFrame(vecs)
        dfres.to_csv('laporan-keuangan-2018-training-data.csv',sep=";", index=False)
    except Exception as e:
        print("Error:",str(e))


def train_knn(training_data_file):
    train=pd.read_csv(training_data_file,sep=";")
    knn = KNeighborsClassifier(n_neighbors=train.shape[0])
    data=train.iloc[:,:-1]
    label=train.iloc[:,-1]
    knn_index={}
    for idx,val in label.iteritems():
        knn_index[idx]=val

    knn.fit(data, label)
    with open("kemkeu_knn_model.pickle","wb") as fi:
        pickle.dump(knn,fi)
        print("train KNN DONE. Model saved in ")
    with open('kemkeu_knn_index_path.pickle',"wb") as fi:
        pickle.dump(knn_index,fi)
        print("Index saved in ")


def id2details_df(df, id_array):
    mask = df['id'].isin(id_array)
    return df.loc[mask]

def get_paragraph(query, n=5):
    original_data_df = pd.read_csv('tfidf_docs/laporan-keuangan-2018-paragraph.csv')

    with open('tfidf_docs/tfidf_model_path.pickle', "rb") as fo:
        fitted_tfidf_vectorizer=pickle.load(fo)
        print("LOADED tfidf model2 from ")

    with open('tfidf_docs/kemkeu_knn_index_path.pickle',"rb") as fi: 
        knn_index=pickle.load(fi)
        print("LOADED KNN index from ")

    with open('tfidf_docs/kemkeu_knn_model.pickle',"rb") as fi: 
        knn_model=pickle.load(fi)
        print("LOADED KNN model from ")

    query=str(query).lower()
    query=remove_punctuation(query)
    query=remove_stopwords(query)
    query=stem_text(query)
    query=remove_english_stopwords(query)
    query=stem_english_text(query)
    query=remove_single_char(query)

    densematrix = fitted_tfidf_vectorizer.transform([query])
    skillvecs=densematrix.toarray()
    vector=np.array(skillvecs[0]).astype('float32')
    vector=np.expand_dims(vector,0)
    (distances, indices) = knn_model.kneighbors(vector, n)
    indices = indices.tolist()
    res=[knn_index[x] for x in indices[0]]
    print(distances)
    result_id = np.array(res).astype('int32')

    return id2details_df(original_data_df, result_id)

#Clean Text
# clean_text()
# Generate training data and tfidf model
# train_tfidfvactorizer()

#Train knn with generated training data
# train_knn("/Users/admin/Code/MINE/hackathon/bpk_2021/laporan-keuangan-2018-training-data.csv")

#get paragraph
# print(get_paragraph("akuntabilitas keuangan negara", 10))
