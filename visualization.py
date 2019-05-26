#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:34:48 2019

@author: salihemredevrim
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#%%
#take datasets 
hip_hop = pd.read_excel('hip_hop_last.xlsx').reset_index(drop=True)

#%%
#PLOTTING *****************************************************************************************************************************
#For yearly word counts per genre and pos

def word_counts(data1, pos, year, most_num):
   
    #most_num: most common x words per year
    #Year is for year column name
    #init
    freq = pd.DataFrame()
    common_words = []
    years = data1[year].unique().tolist()
    
    #frequencies per each year
    for i in range (0, len(years)):
        year_corpus = str(data1[pos][data1[year] == years[i]].tolist())
        #print(year_corpus)
        year_corpus = year_corpus.replace("'", '')
        year_corpus = year_corpus.replace("[", '')
        year_corpus = year_corpus.replace("]", '')
        year_corpus = year_corpus.replace(",", '')
        year_corpus = year_corpus.replace("  ", ' ')
        year_corpus = year_corpus.replace(" nan", '')
        year_corpus = year_corpus.replace("nan", '')
        tokens = year_corpus.split(" ")
        tokens = filter(None, tokens) 
        counts = Counter(tokens)
        freq = freq.append({
            year: years[i],
            "words": counts.most_common(n=most_num)
        }, ignore_index=True)
    freq[year] = freq[year].astype(int)
    
    #distinct words through years 
    for i in range (0, len(freq)): 
        for words in freq['words'][i]:
            common_words.append(words[0])
            
    common_words = list(set(common_words))
    
    #tabularize
    data2 = pd.DataFrame(dict.fromkeys(common_words, [0]))
    data2[year] = 0
    tabularized = data2.copy()
    
    for j in freq[year]:
        row1 = data2.copy()
        row1[year] = j 
        tabularized = tabularized.append(row1)
    
    tabularized = tabularized[1:]
    tabularized = tabularized.reset_index(drop=True)    
    
    
    for j in range(0, len(tabularized)):
            current_year = freq[year][j]
            current_terms = freq['words'][j]
            
            for words in current_terms:
                tabularized[words[0]] = tabularized.apply(lambda x: words[1] if x[year] == current_year else x[words[0]], axis=1)
    
    tabularized_t = tabularized.T.reset_index(drop=False)   
    tabularized_t.columns = tabularized_t.iloc[-1]  
    tabularized_t.drop(tabularized_t.tail(1).index,inplace=True) 
    
    writer = pd.ExcelWriter('For_animated_bars_'+pos+'.xlsx', engine='xlsxwriter');
    tabularized_t.to_excel(writer, sheet_name= 'stream_me');
    writer.save();
    
    return freq, common_words, tabularized

#%%
#How can write labels inside like examples in slides?! Answer is go Tableaueua
#NOT USING THIS

def plotting(tabularized1, year, num_words, genre):
    
    #make year index and take most popular n words
    data3 = tabularized1.set_index(year).sort_index(axis=0, ascending=True)
    sum_over = data3.sum(axis = 0, skipna = True).reset_index(drop=False).sort_values(0, ascending=False).reset_index(drop=True)
    
    #overall top x
    data4 = sum_over.head(num_words) 
    keep_list = data4['index'].tolist()
    data5 = data3[keep_list] 
    
    #ignore 2015 for hip hop   
    if genre == 'hip_hop': 
        data5 = data5[data5.index != 2015]

    #plot
    plt.style.use('seaborn')
    data5.plot.area()
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Frequent Words', fontsize=15)
    plt.title('Most Frequent Words Through Years',fontsize=15)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
    #ax.legend(handles, labels)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    return;

#%%
#For Tableaueau

def go_for_tableau(tabularized1, data1, genre, pos, pos_column, year_name, top_x_word_per_year):
    
    tabularized2 = tabularized1.copy()
    
    list1 = list(tabularized2).remove(year_name)
    sum_all = pd.melt(tabularized2, id_vars=year_name, value_vars=list1)
    sum_all1 = sum_all.groupby(year_name)['value'].sum().reset_index(drop=False)
        
    xxd11 = pd.melt(tabularized2, id_vars=year_name, value_vars=list1)
    xxd22 = xxd11.sort_values([year_name, 'value'], ascending=[True, False])
    xxd33 = xxd22.groupby(year_name).head(top_x_word_per_year)
    
    #take distinct words and top 
    dist_vars = xxd33.variable.unique().tolist()
    
    xxd44 = xxd22[xxd22['variable'].isin(dist_vars)]
    
    #find proportions
    merge1 = pd.merge(xxd44,  sum_all1, how='left', on=year_name)
    
    merge1['word_perc'] = merge1.apply(lambda x: (100*x['value_x']/x['value_y']) if x['value_y'] > 0 else 0, axis=1)
    
    writer = pd.ExcelWriter(genre+'_'+pos+'_stream_me.xlsx', engine='xlsxwriter');
    merge1.to_excel(writer, sheet_name= 'stream_me');
    writer.save();
    
    return merge1;

#%%

#Emotions and sentiments 
emotions = pd.read_csv('hip_hop_nocontracted_v4_emotions.csv')

list1 = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']   
emotions1 = pd.melt(emotions, id_vars='Year', value_vars=list1)

writer = pd.ExcelWriter('Emotions_stream_me.xlsx', engine='xlsxwriter');
emotions1.to_excel(writer, sheet_name= 'stream_me');
writer.save();

#sentiments = pd.read_csv('hip_hop_nocontracted_v4_emotions.csv')
#
#list2 = ['Sentiment_score']   
#sentiments1 = pd.melt(sentiments, id_vars='Year', value_vars=list2)
#
#sentiments1['Pos_Neg'] = sentiments1['value'].apply(lambda x: 'Negative' if x < 0 else 'Positive')
#
#writer = pd.ExcelWriter('Sentiments_stream_me.xlsx', engine='xlsxwriter');
#sentiments1.to_excel(writer, sheet_name= 'stream_me');
#writer.save();

#%%
#Word clouds
def wordcloud(freq, year_name, year, pos, max_words, genre):
    
    
    freq1 = freq.copy()
    #no need to yearly wordclouds anymore
    #freq1 = freq[freq[year_name] == year].reset_index(drop=True)
    #freq2 = pd.DataFrame(freq1['words'][0]).astype(str)
    #freq2 = freq2.rename(index=str, columns={0: 'word', 1: 'count'})
    
    mergedlist = [];
    #for aggregated 
    for k in range(len(freq1)): 
        mergedlist = mergedlist + freq1['words'].iloc[k]
        
    freq2 = pd.DataFrame(mergedlist, columns=['word', 'count'])
    
    freq3 = freq2.groupby('word')['count'].sum().reset_index(drop=False)
      
    d = {}
    for a, x in freq3.values:
        d[a] = float(x)

    wordcloud = WordCloud( width = 4000,
                          height = 3000,
                          background_color="white",
                          max_words = max_words )
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    
    #plt.savefig('WC_'+genre+'_'+pos+'_'+str(year), bbox_inches='tight')
    plt.savefig('WC_'+genre+'_'+pos+'_all', bbox_inches='tight')
    
    return;

#%%
#Heatmap 

def heatmappp(tabularized1, year_name, pos, heat_map_words, genre): 
    
    #make year index and take most popular n words
    heatmap = tabularized1.set_index(year_name)
    sum_over = heatmap.sum(axis = 0, skipna = True).reset_index(drop=False).sort_values(0, ascending=False).reset_index(drop=True)
    heatmap2 = sum_over.head(heat_map_words) 
    keep_list = heatmap2['index'].tolist()
    heatmap3 = heatmap[keep_list] 

    plt.figure(figsize=(10,10))
    sns.heatmap(data=heatmap3, cmap="YlGnBu")
    #plt.show()  
    
    plt.savefig('HM_'+genre+'_'+pos, bbox_inches='tight')
    
    return; 

#%%
#Heatmap and tableau for genre and pos/ner

def run_per_genre(data1, pos, pos_column, genre, year_name, most_num, top_x_word_per_year, heat_map_words, max_words):
    
    #pos: Nouns, verbs, also NERs etc. 
    #genre: hip_hop, rock etc.
    #year_name: 'Year'
    #most_num: taking most repeated x words as columns
    #top_x_word_per_year: for tableau
    #heat_map_words: number of words in heatmap
    #max_words: for wordclouds
    
    frequencies, common_words, tabularized = word_counts(data1, pos, year_name, most_num);
    go_for_tableau(tabularized, data1, genre, pos, pos_column, year_name, top_x_word_per_year);
    heatmappp(tabularized, year_name, pos, heat_map_words, genre);
      
    #wordclouds 
    #for k in frequencies[year_name]:
    #    wordcloud(frequencies, year_name, k, pos, max_words, genre)
    
    wordcloud(frequencies, year_name, 1905, pos, max_words, genre)
    
    return frequencies, common_words, tabularized; 

#%%
#Wordclouds 

frequencies, common_words, tabularized = run_per_genre(hip_hop, 'ORGANIZATION', 'org_count' ,'hip_hop', 'Year', 900, 5, 10, 20);
frequencies, common_words, tabularized = run_per_genre(hip_hop, 'LOC', 'loc_count' ,'hip_hop', 'Year', 900, 5, 10, 25);
frequencies, common_words, tabularized = run_per_genre(hip_hop, 'DATE', 'date_count' ,'hip_hop', 'Year', 900, 5, 10, 25);
frequencies, common_words, tabularized = run_per_genre(hip_hop, 'PERSON', 'person_count' ,'hip_hop', 'Year', 900, 5, 10, 25);
    
frequencies, common_words, tabularized = run_per_genre(hip_hop, 'Nouns', 'noun_count' ,'hip_hop', 'Year', 900, 10, 10, 25);
frequencies, common_words, tabularized = run_per_genre(hip_hop, 'Verbs', 'verb_count' ,'hip_hop', 'Year', 900, 10, 10, 25);
frequencies, common_words, tabularized = run_per_genre(hip_hop, 'Adjectives', 'adjective_count' ,'hip_hop', 'Year', 900, 10, 10, 25);
frequencies, common_words, tabularized = run_per_genre(hip_hop, 'Adverbs', 'adverb_count' ,'hip_hop', 'Year', 900, 5, 10, 25);


#%%

data222 = pd.read_csv('hip_hop_nocontracted_v4_nostopwords.csv')
frequencies, common_words, tabularized = run_per_genre(data222, 'Lyrics', 'word_count' ,'hip_hop', 'Year', 500, 5, 10, 25);
    
#%%
#singer counts

singers = hip_hop[['Year', 'Artist', 'Song Title']].groupby(['Year', 'Artist'])['Song Title'].count().reset_index(drop=False)

xxd1 = pd.pivot_table(singers, values='Song Title', index='Artist', columns='Year').fillna(0)

writer = pd.ExcelWriter('Artist_stream_me.xlsx', engine='xlsxwriter');
xxd1.to_excel(writer, sheet_name= 'stream_me');
writer.save();
