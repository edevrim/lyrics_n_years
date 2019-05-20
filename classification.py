#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:36:20 2019

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
#take dataset

hip_hop = pd.read_excel('hip_hop_last.xlsx')

#lyrics only pos
hip_hop['lyrics_pos'] = hip_hop['Nouns'].astype(str) + ' ' + hip_hop['Verbs'].astype(str) + ' ' + hip_hop['Adverbs'].astype(str) + ' ' + hip_hop['Adjectives'].astype(str)

#take topics and processed_lyrics
hip_hop_old = pd.read_csv('hip_hop_topic_modeling10.csv')

#
keep1 = ['Year', 'Artist', 'processed_lyrics', 'Song Title', 'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic', 'Topic6', 'Topic7', 'Topic8', 'Topic9', 'Topic10']      
hip_hop_old = hip_hop_old[keep1]

hip_hop_last = pd.merge(hip_hop, hip_hop_old, how='left', on=('Year', 'Artist', 'Song Title'))

lyrics_and_numeric_var_list = ['Year',
'Lyrics',                                
'processed_lyrics',
'lyrics_pos', 
'word_count',
'unique_word_count',
'verb_count',
'noun_count',
'adverb_count',
'adjective_count',
'corpus_count',
'loc_count',
'date_count',
'person_count',
'org_count',
'person_ratio',
'loc_ratio',
'org_ratio',
'unique_ratio',
'verb_to_unique',
'noun_to_unique',
'adj_to_unique',
'adv_to_unique',
'Topic1',
'Topic2',
'Topic3',
'Topic4',
'Topic',
'Topic6',
'Topic7',
'Topic8',
'Topic9',
'Topic10']

hip_hop_last = hip_hop_last[lyrics_and_numeric_var_list]

count1 = hip_hop_last['Year'].value_counts().reset_index(drop=False)

hip_hop1 = hip_hop_last[hip_hop_last['Year'] <= 2007]

hip_hop2 = hip_hop_last[hip_hop_last['Year'] > 2012]

#%%
#Simple rock vs hiphop tfidf etc 
#hiphop: 1 rock: 0 
#also we can try with nouns, verbs etc 

def data_prep(data1, data2, year_name, test_percent):
    #Data1: target 1 data 
    #Data2: target 0 data 
    #text: column for text    
    
    #Data preparation     
    data11 = data1.copy()
    data11['Target'] = 1

    data22 = data2.copy()
    data22['Target'] = 0
    
    #Balance 
    min1 = min(len(data11), len(data22));    
    data11 = data11.sample(n=min1, random_state=1905)
    data22 = data22.sample(n=min1, random_state=1905)
    
    data_all = data11.append(data22, ignore_index=True)
    
    X_train, X_test, y_train, y_test = train_test_split(data_all.drop(['Target', year_name], axis=1), data_all['Target'], stratify=data_all['Target'], test_size=test_percent, random_state=1905)
   
    y_train = pd.DataFrame(y_train).reset_index(drop=True)
    y_test = pd.DataFrame(y_test).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

#%%
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
def log_reg_and_svm(XX_train, YY_train, XX_test, YY_test):     
    
    #Logistic Regression
    model1 = LogisticRegression()
    model1.fit(XX_train, YY_train)

    #Predict on test set
    predictions = model1.predict(XX_test)

    #score
    roc1 = roc_auc_score(YY_test, predictions)
    accuracy1 = accuracy_score(YY_test, predictions)
    precision1 = precision_score(YY_test, predictions)
    recall1 = recall_score(YY_test, predictions)
    f1_score1 = f1_score(YY_test, predictions)
    
    #F1 = 2 * (precision * recall) / (precision + recall)

    #SVM
    model2 = SVC()
    model2.fit(XX_train, YY_train)

    #Predict on test set
    predictions2 = model2.predict(XX_test)

    #score
    roc2 = roc_auc_score(YY_test, predictions2)
    accuracy2 = accuracy_score(YY_test, predictions2)
    precision2 = precision_score(YY_test, predictions2)
    recall2 = recall_score(YY_test, predictions2)
    f1_score2 = f1_score(YY_test, predictions2)
    
    output = {
         'Accuracy LR:': accuracy1,  
         'ROC LR:': roc1,    
         'Precision LR:': precision1,    
         'Recall LR:': recall1,    
         'F1-score LR:': f1_score1,    
         
         'Accuracy SVM:': accuracy2,  
         'ROC SVM:': roc2,    
         'Precision SVM:': precision2,    
         'Recall SVM:': recall2,    
         'F1-score SVM:': f1_score2}
    
    
    return output

#%%
    
def all_techniques(data1, data2, year_name, plain_lyrics, processed_lyrics, pos_lyrics, test_percent, min_df, max_df, ngram_range1, ngram_range2, vector_size1, window1, negative1, min_count1): 

    #Doc2vec parameters: vector_size, window, negative, min_count
    
    #train - test split 
    X_train, X_test, y_train, y_test = data_prep(data1, data2, year_name, test_percent); 
    
    #Simple BOW ********************************************************************************************************
    
    #min_df is used for removing terms that appear too infrequently. For example:
    #min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
    #min_df = 5 means "ignore terms that appear in less than 5 documents".
    #max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
    #max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
    #max_df = 25 means "ignore terms that appear in more than 25 documents".

    #WITH PLAIN LYRICS
    TEXT = plain_lyrics;
    
    #tokenization
    vect = CountVectorizer(max_df=max_df, min_df=min_df).fit(X_train[TEXT])

    #transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train[TEXT])
    X_test_vectorized = vect.transform(X_test[TEXT])

    #models with only lyrics
    output1_bow_plain_lyrics = log_reg_and_svm(X_train_vectorized, y_train, X_test_vectorized, y_test);
    
    #Numeric variables included (Counts and Topics)
    X_train_num = X_train.drop([plain_lyrics, processed_lyrics, pos_lyrics], axis=1)
    X_test_num = X_test.drop([plain_lyrics, processed_lyrics, pos_lyrics], axis=1)
    
    dummy1 = pd.DataFrame(X_train_vectorized.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())

    #models 
    output2_bow_plain_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);

    #Tf-Idf ******************************************************************************************************************

    vect_tf = TfidfVectorizer(min_df= min_df, max_df= max_df).fit(X_train[TEXT])

    #transform
    X_train_vectorized_tf = vect_tf.transform(X_train[TEXT])
    X_test_vectorized_tf = vect_tf.transform(X_test[TEXT])
    
    #models with only lyrics
    output3_tfidf_plain_lyrics = log_reg_and_svm(X_train_vectorized_tf, y_train, X_test_vectorized_tf, y_test);
        
    #Numeric variables included (Counts and Topics)

    dummy1 = pd.DataFrame(X_train_vectorized_tf.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized_tf.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())
    
    #models 
    output4_tfidf_plain_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);


    #N-grams ******************************************************************************************************************
    
    #document frequency of 5 and extracting 1-grams and 2-grams...
    vect3 = CountVectorizer(min_df=min_df, ngram_range=(ngram_range1, ngram_range2)).fit(X_train[TEXT])
    
    X_train_vectorized_ng = vect3.transform(X_train[TEXT])
    X_test_vectorized_ng = vect3.transform(X_test[TEXT])
    
    #models with only lyrics
    output5_ngram_plain_lyrics = log_reg_and_svm(X_train_vectorized_ng, y_train, X_test_vectorized_ng, y_test);
        
    #Numeric variables included (Counts and Topics)

    dummy1 = pd.DataFrame(X_train_vectorized_ng.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized_ng.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())
    
    #models 
    output6_ngram_plain_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);

 
    #Doc2Vec ********************************************************************************************************************************************************
    
    train_doc = pd.DataFrame(pd.concat([X_train[TEXT], y_train], axis=1))
    test_doc = pd.DataFrame(pd.concat([X_test[TEXT], y_test], axis=1))
    
    train_doc2 = train_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x['Target']]), axis=1)
    test_doc2 = test_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x['Target']]), axis=1)

    #DBOW is the Doc2Vec model analogous to Skip-gram model in Word2Vec. 
    #The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.
    #We set the minimum word count to 2 in order to discard words with very few occurrences.

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0,  vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, hs=0, workers=cores)
    model_dbow.build_vocab([x for x in train_doc2.values])
    
    y_train_doc, X_train_doc = get_vectors(model_dbow, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dbow, test_doc2)
    
    #models 
    output7_doc2vec_dbow_plain_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);
 
    #dm = 1 model
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, workers=5)
    model_dmm.build_vocab([x for x in train_doc2.values])

    for epoch in range(100):
        model_dmm.train(utils.shuffle([x for x in train_doc2.values]), total_examples=len(train_doc2.values), epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha
    
    y_train_doc, X_train_doc = get_vectors(model_dmm, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dmm, test_doc2)
    
    #models 
    output8_doc2vec_dm_plain_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);

    #Mix of dbow and dmm
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    
    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    
    y_train_doc, X_train_doc = get_vectors(new_model, train_doc2)
    y_test_doc, X_test_doc = get_vectors(new_model, test_doc2)
    
    #models 
    output9_doc2vec_dbowdb_plain_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);


###############################################################################################################
    
    #WITH PROCESSED LYRICS
    TEXT = processed_lyrics;
    
    #tokenization
    vect = CountVectorizer(max_df=max_df, min_df=min_df).fit(X_train[TEXT])

    #transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train[TEXT])
    X_test_vectorized = vect.transform(X_test[TEXT])

    #models with only lyrics
    output10_bow_processed_lyrics = log_reg_and_svm(X_train_vectorized, y_train, X_test_vectorized, y_test);
    
    #Numeric variables included (Counts and Topics)
    X_train_num = X_train.drop([plain_lyrics, processed_lyrics, pos_lyrics], axis=1)
    X_test_num = X_test.drop([plain_lyrics, processed_lyrics, pos_lyrics], axis=1)
    
    dummy1 = pd.DataFrame(X_train_vectorized.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())

    #models 
    output11_bow_processed_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);

    #Tf-Idf ******************************************************************************************************************

    vect_tf = TfidfVectorizer(min_df= min_df, max_df= max_df).fit(X_train[TEXT])

    #transform
    X_train_vectorized_tf = vect_tf.transform(X_train[TEXT])
    X_test_vectorized_tf = vect_tf.transform(X_test[TEXT])
    
    #models with only lyrics
    output12_tfidf_processed_lyrics = log_reg_and_svm(X_train_vectorized_tf, y_train, X_test_vectorized_tf, y_test);
        
    #Numeric variables included (Counts and Topics)

    dummy1 = pd.DataFrame(X_train_vectorized_tf.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized_tf.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())
    
    #models 
    output13_tfidf_processed_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);


    #N-grams ******************************************************************************************************************
    
    #document frequency of 5 and extracting 1-grams and 2-grams...
    vect3 = CountVectorizer(min_df=min_df, ngram_range=(ngram_range1, ngram_range2)).fit(X_train[TEXT])
    
    X_train_vectorized_ng = vect3.transform(X_train[TEXT])
    X_test_vectorized_ng = vect3.transform(X_test[TEXT])
    
    #models with only lyrics
    output14_ngram_processed_lyrics = log_reg_and_svm(X_train_vectorized_ng, y_train, X_test_vectorized_ng, y_test);
        
    #Numeric variables included (Counts and Topics)

    dummy1 = pd.DataFrame(X_train_vectorized_ng.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized_ng.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())
    
    #models 
    output15_ngram_processed_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);
 
    #Doc2Vec ********************************************************************************************************************************************************
    
    train_doc = pd.DataFrame(pd.concat([X_train[TEXT], y_train], axis=1))
    test_doc = pd.DataFrame(pd.concat([X_test[TEXT], y_test], axis=1))
    
    train_doc2 = train_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x['Target']]), axis=1)
    test_doc2 = test_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x['Target']]), axis=1)

    #DBOW is the Doc2Vec model analogous to Skip-gram model in Word2Vec. 
    #The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.
    #We set the minimum word count to 2 in order to discard words with very few occurrences.

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0,  vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, hs=0, workers=cores)
    model_dbow.build_vocab([x for x in train_doc2.values])
    
    y_train_doc, X_train_doc = get_vectors(model_dbow, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dbow, test_doc2)
    
    #models 
    output16_doc2vec_dbow_processed_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);
 
    #dm = 1 model
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, workers=5)
    model_dmm.build_vocab([x for x in train_doc2.values])

    for epoch in range(100):
        model_dmm.train(utils.shuffle([x for x in train_doc2.values]), total_examples=len(train_doc2.values), epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha
    
    y_train_doc, X_train_doc = get_vectors(model_dmm, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dmm, test_doc2)
    
    #models 
    output17_doc2vec_dm_processed_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);

    #Mix of dbow and dmm
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    
    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    
    y_train_doc, X_train_doc = get_vectors(new_model, train_doc2)
    y_test_doc, X_test_doc = get_vectors(new_model, test_doc2)
    
    #models 
    output18_doc2vec_dbowdb_processed_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);


######################################################################################################################


    #WITH POS LYRICS
    TEXT = pos_lyrics;
    
    #tokenization
    vect = CountVectorizer(max_df=max_df, min_df=min_df).fit(X_train[TEXT])

    #transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train[TEXT])
    X_test_vectorized = vect.transform(X_test[TEXT])

    #models with only lyrics
    output19_bow_pos_lyrics = log_reg_and_svm(X_train_vectorized, y_train, X_test_vectorized, y_test);
    
    #Numeric variables included (Counts and Topics)
    X_train_num = X_train.drop([plain_lyrics, processed_lyrics, pos_lyrics], axis=1)
    X_test_num = X_test.drop([plain_lyrics, processed_lyrics, pos_lyrics], axis=1)
    
    dummy1 = pd.DataFrame(X_train_vectorized.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())

    #models 
    output20_bow_pos_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);

    #Tf-Idf ******************************************************************************************************************

    vect_tf = TfidfVectorizer(min_df= min_df, max_df= max_df).fit(X_train[TEXT])

    #transform
    X_train_vectorized_tf = vect_tf.transform(X_train[TEXT])
    X_test_vectorized_tf = vect_tf.transform(X_test[TEXT])
    
    #models with only lyrics
    output21_tfidf_pos_lyrics = log_reg_and_svm(X_train_vectorized_tf, y_train, X_test_vectorized_tf, y_test);
        
    #Numeric variables included (Counts and Topics)

    dummy1 = pd.DataFrame(X_train_vectorized_tf.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized_tf.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())
    
    #models 
    output22_tfidf_pos_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);


    #N-grams ******************************************************************************************************************
    
    #document frequency of 5 and extracting 1-grams and 2-grams...
    vect3 = CountVectorizer(min_df=min_df, ngram_range=(ngram_range1, ngram_range2)).fit(X_train[TEXT])
    
    X_train_vectorized_ng = vect3.transform(X_train[TEXT])
    X_test_vectorized_ng = vect3.transform(X_test[TEXT])
    
    #models with only lyrics
    output23_ngram_pos_lyrics = log_reg_and_svm(X_train_vectorized_ng, y_train, X_test_vectorized_ng, y_test);
        
    #Numeric variables included (Counts and Topics)

    dummy1 = pd.DataFrame(X_train_vectorized_ng.toarray())
    dummy2 = pd.DataFrame(X_test_vectorized_ng.toarray())

    X_train_mixed = pd.concat([dummy1, X_train_num], axis=1)
    X_test_mixed = pd.concat([dummy2, X_test_num], axis=1)
    
    #fill na with train means
    X_train_mixed = X_train_mixed.fillna(X_train_mixed.mean())
    X_test_mixed = X_test_mixed.fillna(X_train_mixed.mean())
    
    #models 
    output24_ngram_pos_lyrics_w_num = log_reg_and_svm(X_train_mixed, y_train, X_test_mixed, y_test);
 
    #Doc2Vec ********************************************************************************************************************************************************
    
    train_doc = pd.DataFrame(pd.concat([X_train[TEXT], y_train], axis=1))
    test_doc = pd.DataFrame(pd.concat([X_test[TEXT], y_test], axis=1))
    
    train_doc2 = train_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x['Target']]), axis=1)
    test_doc2 = test_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[TEXT]), tags=[x['Target']]), axis=1)

    #DBOW is the Doc2Vec model analogous to Skip-gram model in Word2Vec. 
    #The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.
    #We set the minimum word count to 2 in order to discard words with very few occurrences.

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0,  vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, hs=0, workers=cores)
    model_dbow.build_vocab([x for x in train_doc2.values])
    
    y_train_doc, X_train_doc = get_vectors(model_dbow, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dbow, test_doc2)
    
    #models 
    output25_doc2vec_dbow_pos_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);
 
    #dm = 1 model
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, workers=5)
    model_dmm.build_vocab([x for x in train_doc2.values])

    for epoch in range(100):
        model_dmm.train(utils.shuffle([x for x in train_doc2.values]), total_examples=len(train_doc2.values), epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha
    
    y_train_doc, X_train_doc = get_vectors(model_dmm, train_doc2)
    y_test_doc, X_test_doc = get_vectors(model_dmm, test_doc2)
    
    #models 
    output26_doc2vec_dm_pos_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);

    #Mix of dbow and dmm
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    
    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    
    y_train_doc, X_train_doc = get_vectors(new_model, train_doc2)
    y_test_doc, X_test_doc = get_vectors(new_model, test_doc2)
    
    #models 
    output27_doc2vec_dbowdb_pos_lyrics = log_reg_and_svm(X_train_doc, y_train_doc, X_test_doc, y_test_doc);


    xxd11 = pd.DataFrame(output1_bow_plain_lyrics, index=['bow_plain_lyrics'])
    xxd11 = xxd11.append(pd.DataFrame(output2_bow_plain_lyrics_w_num, index=['bow_plain_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output3_tfidf_plain_lyrics, index=['tfidf_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output4_tfidf_plain_lyrics_w_num, index=['tfidf_plain_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output5_ngram_plain_lyrics, index=['ngram_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output6_ngram_plain_lyrics_w_num, index=['ngram_plain_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output7_doc2vec_dbow_plain_lyrics, index=['doc2vec_dbow_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output8_doc2vec_dm_plain_lyrics, index=['doc2vec_dm_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output9_doc2vec_dbowdb_plain_lyrics, index=['doc2vec_dbowdb_plain_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output10_bow_processed_lyrics, index=['bow_processed_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output11_bow_processed_lyrics_w_num, index=['bow_processed_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output12_tfidf_processed_lyrics, index=['tfidf_processed_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output13_tfidf_processed_lyrics_w_num, index=['tfidf_processed_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output14_ngram_processed_lyrics, index=['ngram_processed_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output15_ngram_processed_lyrics_w_num, index=['ngram_processed_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output16_doc2vec_dbow_processed_lyrics, index=['doc2vec_dbow_processed_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output17_doc2vec_dm_processed_lyrics, index=['doc2vec_dm_processed_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output18_doc2vec_dbowdb_processed_lyrics, index=['doc2vec_dbowdb_processed_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output19_bow_pos_lyrics, index=['bow_pos_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output20_bow_pos_lyrics_w_num, index=['bow_pos_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output21_tfidf_pos_lyrics, index=['tfidf_pos_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output22_tfidf_pos_lyrics_w_num, index=['tfidf_pos_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output23_ngram_pos_lyrics, index=['ngram_pos_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output24_ngram_pos_lyrics_w_num, index=['ngram_pos_lyrics_w_num']))
    xxd11 = xxd11.append(pd.DataFrame(output25_doc2vec_dbow_pos_lyrics, index=['doc2vec_dbow_pos_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output26_doc2vec_dm_pos_lyrics, index=['doc2vec_dm_pos_lyrics']))
    xxd11 = xxd11.append(pd.DataFrame(output27_doc2vec_dbowdb_pos_lyrics, index=['doc2vec_dbowdb_pos_lyrics']))

    return xxd11

#%%
   
output1 = all_techniques(hip_hop1, hip_hop2, 'Year', 'Lyrics', 'processed_lyrics', 'lyrics_pos', 0.3, 3, 0.95, 1, 3, 300, 10, 5, 3) 
#all_techniques(data1, data2, year_name, plain_lyrics, processed_lyrics, pos_lyrics, test_percent, min_df, max_df, ngram_range1, ngram_range2, vector_size1, window1, negative1, min_count1): 
