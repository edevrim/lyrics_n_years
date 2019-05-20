#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:04:11 2019

@author: salihemredevrim
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from nltk.tag import StanfordNERTagger

#%%
#DATA PREP *****************************************************************************************************************************

#datasets
hip_hop_no_lc = pd.read_csv('hip_hop_nocontracted_v4_upper.csv')
hip_hop = pd.read_csv('hip_hop_topic_modeling10.csv')

keep_list = ['Year', 'Artist', 'Song Title', 'Lyrics', 'word_count', 'processed_lyrics', 'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic', 'Topic6', 'Topic7', 'Topic8', 'Topic9', 'Topic10']

#%%
#Number of songs, average word count and average unique count per year

def some_analyses(data1, genre, word_cutoff, top_singers):
    
    #check duplicates
    duplicates = data1.Lyrics.value_counts().reset_index(drop=False)
    duplicates = duplicates[duplicates['Lyrics'] > 1]

    data1['counter'] = data1.groupby('Lyrics').cumcount() + 1

    #take the first version
    data1 = data1[data1['counter'] == 1]

    #outlier elimination 
    data1 = data1[data1['word_count'] <= word_cutoff]
    
    data1 = data1[keep_list]
    
    #unique words
    # Split list into new series
    lyrics = data1['Lyrics'].str.split()

    # Get amount of unique words
    data1['unique_word_count'] = lyrics.apply(set).apply(len)

    count1 = data1.groupby('Year')['Lyrics'].count().reset_index(drop=False)
    count2 = data1.groupby('Artist')['Lyrics'].count().reset_index(drop=False).sort_values('Lyrics', ascending=False).head(top_singers)
    count3 = data1.groupby('Year')['word_count'].mean().reset_index(drop=False)
    count4 = data1.groupby('Year')['unique_word_count'].mean().reset_index(drop=False)

    sns.barplot(y=count1.Lyrics, x=count1.Year)
    plt.title('Number of Songs per Year')
    plt.ylabel('Number of Songs')
    plt.xticks(rotation=45)
    #plt.show()
    
    plt.savefig('Number of Songs per Year_'+genre, bbox_inches='tight')
    
    sns.barplot(y=count3.word_count, x=count3.Year)
    plt.title('Average Words Count per Year')
    plt.ylabel('Word Counts')
    plt.xticks(rotation=45)
    #plt.show()
    
    plt.savefig('Average Words Count per Year_'+genre, bbox_inches='tight')
    
    sns.barplot(y=count4.unique_word_count, x=count4.Year)
    plt.title('Average Unique Words Count per Year')
    plt.ylabel('Unique Words Counts')
    plt.xticks(rotation=45)
    #plt.show()
    
    plt.savefig('Average Unique Words Count per Year_'+genre, bbox_inches='tight')
    
    #stacked all 
    df1 = pd.merge(count1, count3, how='left', on='Year')
    df1 = pd.merge(df1, count4, how='left', on='Year')

    # create a color palette
    palette = plt.get_cmap('Set1')

    plt.style.use('seaborn-darkgrid')
    my_dpi=96
    plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
    num = 1; 
    
    # multiple line plot
    for column in df1.drop('Year', axis=1):
        num+=1
        plt.plot(df1['Year'], df1[column], marker='', color=palette(num), linewidth=3, alpha=0.4, label=column)
 
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Add titles
    plt.title("Average Word, Unique Word and Number of Lyrics Counts per Year", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Year")
    plt.ylabel("Count")
    
    plt.savefig('Average Word, Unique Word and Number of Lyrics Counts per Year_'+genre, bbox_inches='tight')
    
    return data1, count2

#%%
#find verbs, adverbs, nouns...

def spacy_data(data1, lyrics):

    #init 
    verbs = []
    nouns = []
    adverbs = []
    adj = []
    corpus = []
    nlp = spacy.load('en_core_web_md')
    
    for i in range (0, len(data1)):
        #print('song', i)
        song = data1.iloc[i][lyrics]
        doc = nlp(song)
        spacy_data = pd.DataFrame()
        
        for token in doc:
            if token.lemma_ == "-PRON-":
                    lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "word": token.text,
                "lemma": lemma,
                "pos": token.pos_,
                "stop_word": token.is_stop
            }  
            spacy_data = spacy_data.append(row, ignore_index = True)
        
        adj.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "ADJ"].values))
        verbs.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "VERB"].values))
        nouns.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "NOUN"].values))
        adverbs.append(" ".join(spacy_data["lemma"][spacy_data["pos"] == "ADV"].values))
        corpus_clean = " ".join(spacy_data["lemma"][spacy_data["stop_word"] == False].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   
        corpus.append(corpus_clean)
        
    data1['Verbs'] = verbs
    data1['Nouns'] = nouns
    data1['Adverbs'] = adverbs
    data1['Adjectives'] = adj
    data1['Corpus'] = corpus
    
    #unique words
    # Split list into new series
    verbs1 = data1['Verbs'].str.split()
    nouns1 = data1['Nouns'].str.split()
    adverbs1 = data1['Adverbs'].str.split()
    adjectives1 = data1['Adjectives'].str.split()
    corpus1 = data1['Corpus'].str.split()
    
    # Get amount of unique words
    data1['verb_count'] = verbs1.apply(set).apply(len)
    data1['noun_count'] = nouns1.apply(set).apply(len)
    data1['adverb_count'] = adverbs1.apply(set).apply(len)
    data1['adjective_count'] = adjectives1.apply(set).apply(len)
    data1['corpus_count'] = corpus1.apply(set).apply(len)
    
    return data1

#%%    
#NOT USING THIS! 
#NER by Spacy 
def ner_spacy(data1, lyrics):

#case sensitive, use the data without lower etc.
#https://medium.com/@dudsdu/named-entity-recognition-for-unstructured-documents-c325d47c7e3a   
   
    nlp = spacy.load('en')
    
    #PERSON	People, including fictional.
    #NORP	Nationalities or religious or political groups.
    #FAC	Buildings, airports, highways, bridges, etc.
    #ORG	Companies, agencies, institutions, etc.
    #GPE	Countries, cities, states.
    #LOC	Non-GPE locations, mountain ranges, bodies of water.
    #PRODUCT	Objects, vehicles, foods, etc. (Not services.)
    #EVENT	Named hurricanes, battles, wars, sports events, etc.
    #WORK_OF_ART	Titles of books, songs, etc.
    
    #merged (person, norp, fac, org, product, work_of_art, event) and (loc, gpe) and (date)
    data1['LOC'] = ''; 
    data1['DATE'] = '';
    data1['OTHERS'] = '';
     
    for i in range(0, len(data1)):

        loc = '';
        others = '';
        date = '';
        
        song = data1.iloc[i][lyrics]
        doc = nlp(song)
        ents = [(e.text, e.label_) for e in doc.ents]
        ents1 = pd.DataFrame(ents)
        
        for k in range(0, len(ents1)):
            
            if ents1.iloc[k][1] in ('PERSON', 'NORP','FAC', 'PRODUCT', 'ORG', 'WORK_OF_ART', 'EVENT'):   
              others = others+ ', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] in ('LOC', 'GPE'):
              loc =  loc+', '+ents1.iloc[k][0]
            elif ents1.iloc[k][1] == 'DATE':
                 date =  date+', '+ents1.iloc[k][0]
                 
        data1['LOC'].iloc[i] = loc;
        data1['DATE'].iloc[i] = date;
        data1['OTHERS'].iloc[i] = others;
    
    return data1

#%%
#NER by Stanford   

def ner_stanford(data1, lyrics): 
    #print('NTLK Version: %s' % nltk.__version__)
    stanford_ner_tagger = StanfordNERTagger(
    'stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    'stanford_ner/' + 'stanford-ner-3.9.2.jar')
      
    data1['LOC'] = ''; 
    data1['DATE'] = '';
    data1['PERSON'] = '';
    data1['ORGANIZATION'] = '';
     
    for i in range(0, len(data1)):
        
        loc = '';
        date = '';
        org = '';
        person = '';
        prev_tag_type = 'HELLO'
        prev_tag_value = 'HELLO'
        
        song = data1.iloc[i][lyrics]
        results = stanford_ner_tagger.tag(song.split())
        
        for result in results:
           
            tag_value = result[0]
            tag_type = result[1]
            
            if tag_type == 'PERSON': 
                if prev_tag_type == 'PERSON':
                    person = person+ ' '+tag_value
                else:
                    person = person+ ', '+tag_value
                    
            elif tag_type == 'LOCATION':
                if prev_tag_type == 'LOCATION':
                    loc =  loc+' '+tag_value
                else:
                    loc =  loc+', '+tag_value
                    
            elif (tag_type == 'ORGANIZATION' and prev_tag_value != tag_value): 
                if prev_tag_type == 'ORGANIZATION':
                    org = org+ ' '+tag_value
                else:
                    org = org+ ', '+tag_value
                    
            #No need for date        
            elif tag_type == 'DATE': 
              date = date+ ', '+tag_value
              
            prev_tag_type = tag_type  
            prev_tag_value = tag_value 
              
              
        data1['LOC'].iloc[i] = loc;
        data1['DATE'].iloc[i] = date;
        data1['PERSON'].iloc[i] = person;
        data1['ORGANIZATION'].iloc[i] = org;
        
        #unique words
        # Split list into new series
        loc1 = data1['LOC'].str.split(',')
        date1 = data1['DATE'].str.split(',')
        person1 = data1['PERSON'].str.split(',')
        org1 = data1['ORGANIZATION'].str.split(',')
    
        # Get amount of unique words
        data1['loc_count'] = loc1.apply(set).apply(len) - 1
        data1['date_count'] = date1.apply(set).apply(len) - 1 
        data1['person_count'] = person1.apply(set).apply(len) - 1
        data1['org_count'] = org1.apply(set).apply(len) - 1
 
    return data1  
    

#%%
#All together 
#Creates all dataset

def create_my_data(data_lower, data_non_lower, lyrics, genre, word_cutoff, top_singers):
    
    data1, count1 = some_analyses(data_lower, genre, word_cutoff, top_singers)
    
    data1 = spacy_data(data1, lyrics); 
    data2 = ner_stanford(data_non_lower, lyrics); 
    
    keep1 = ['Year','Artist','Lyrics','Song Title','Verbs','Nouns','Adverbs','Adjectives', 'Corpus','word_count','unique_word_count','verb_count', 'noun_count','adverb_count','adjective_count' ,'corpus_count']
    keep2 = ['LOC', 'DATE', 'PERSON', 'ORGANIZATION', 'loc_count', 'date_count', 'person_count', 'org_count']
    
    data3 = pd.merge(data1[keep1], data2[keep2], how='left', left_index=True, right_index=True)
    
    data3['person_ratio'] = data3.apply(lambda x: x['person_count']/ x['noun_count'] if x['noun_count'] > 0 else 0, axis=1)
    data3['loc_ratio'] = data3.apply(lambda x: x['loc_count']/ x['noun_count'] if x['noun_count'] > 0 else 0, axis=1)
    data3['org_ratio'] = data3.apply(lambda x: x['org_count']/ x['noun_count'] if x['noun_count'] > 0 else 0, axis=1)
    data3['unique_ratio'] = data3.apply(lambda x: x['unique_word_count']/ x['word_count'] if x['word_count'] > 0 else 0, axis=1)
    
    data3['verb_to_unique'] = data3.apply(lambda x: x['verb_count']/ x['unique_word_count'] if x['unique_word_count'] > 0 else 0, axis=1)
    data3['noun_to_unique'] = data3.apply(lambda x: x['noun_count']/ x['unique_word_count'] if x['unique_word_count'] > 0 else 0, axis=1)
    data3['adj_to_unique'] = data3.apply(lambda x: x['adjective_count']/ x['unique_word_count'] if x['unique_word_count'] > 0 else 0, axis=1)
    data3['adv_to_unique'] = data3.apply(lambda x: x['adverb_count']/ x['unique_word_count'] if x['unique_word_count'] > 0 else 0, axis=1)
    
    #for visualization let's make it like New-York
    data3['LOC'] = data3['LOC'].apply(lambda x: x.replace(' ', '-'))
    data3['LOC'] = data3['LOC'].apply(lambda x: x.replace(',-', ' '))
    
    data3['DATE'] = data3['DATE'].apply(lambda x: x.replace(' ', '-'))
    data3['DATE'] = data3['DATE'].apply(lambda x: x.replace(',-', ' '))
    
    data3['PERSON'] = data3['PERSON'].apply(lambda x: x.replace(' ', '-'))
    data3['PERSON'] = data3['PERSON'].apply(lambda x: x.replace(',-', ' '))
    
    data3['ORGANIZATION'] = data3['ORGANIZATION'].apply(lambda x: x.replace(' ', '-'))
    data3['ORGANIZATION'] = data3['ORGANIZATION'].apply(lambda x: x.replace(',-', ' '))
    
    #to excel
    writer = pd.ExcelWriter(genre+'_last.xlsx', engine='xlsxwriter');
    data3.to_excel(writer, sheet_name= genre);
    writer.save();
    
    return data3, count1

#%%
    
hip_hop_last, hip_hop_top_singers = create_my_data(hip_hop, hip_hop_no_lc, 'Lyrics', 'hip_hop', 2000, 20)    

##to excel
#writer = pd.ExcelWriter(genre+'_last.xlsx', engine='xlsxwriter');
#hip_hop_last.to_excel(writer, sheet_name= genre);
#writer.save();

