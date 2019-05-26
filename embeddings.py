#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:29:57 2019

@author: salihemredevrim
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score,  precision_score, recall_score, f1_score
import nltk
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

#%%

embeddings = pd.read_excel('hip_hop_last.xlsx')

text = 'Lyrics'
embeddings1 = embeddings[['Year', 'Artist', 'Song Title', text]].reset_index(drop=True).reset_index(drop=False)

#%%

#Doc2Vec ********************************************************************************************************************************************************

#Functions for Doc2Vec 
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens  

#%%
def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

#%%

train_doc2 = embeddings1.apply(lambda x: TaggedDocument(words=tokenize_text(x[text]), tags=[x['index']]), axis=1)
   
cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=0,  vector_size=300, window=10, negative=5, min_count=3, hs=0, workers=cores, epochs=200)
train_corpus = [x for x in train_doc2.values]
model_dbow.build_vocab([x for x in train_doc2.values])
model_dbow.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
    
index, vectors = get_vectors(model_dbow, train_doc2)

df = pd.DataFrame(list(vectors)).reset_index(drop=False)

to_tensor_flor_projector = pd.merge(embeddings1[['index', 'Year', 'Artist', 'Song Title']], df, how='left', on='index')


#pick some artist that I know 
count11 = to_tensor_flor_projector['Artist'].value_counts().reset_index(drop=False)

list111 = [
'Drake',
'Beyonce',
'Chris Brown',
'Usher',
'Kanye West',
'Alicia Keys',
'The Weeknd',
'Rihanna',
'50 Cent',
'Robin Thicke',
'Nicki Minaj',
'Eminem',
'Ludacris',
'JAY-Z',
'Mariah Carey',
'Wiz Khalifa',
'John Legend',
'Childish Gambino',
'Justin Timberlake',
'Big Sean',
'Snoop Dogg',
'Bruno Mars',
'Whitney Houston',
"Destiny's Child",
'Sean Paul']

to_tensor_flor_projector2 = to_tensor_flor_projector[to_tensor_flor_projector['Artist'].isin(list111)].reset_index(drop=True)


writer = pd.ExcelWriter('to_tensor_flor_projector.xlsx', engine='xlsxwriter');
to_tensor_flor_projector2.to_excel(writer, sheet_name= 'embeddings');
writer.save();
#%%

    
    