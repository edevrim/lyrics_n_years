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
        tokens = year_corpus.split(" ")
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
    
    count_pos = data1[[pos_column, year_name]]
    count_pos1 = count_pos.groupby(year_name)[pos_column].sum().reset_index(drop=False)
    
    
    list1 = list(tabularized2).remove(year_name)
    xxd11 = pd.melt(tabularized2, id_vars=year_name, value_vars=list1)
    
    xxd22 = xxd11.sort_values([year_name, 'value'], ascending=[True, False])
    
    xxd33 = xxd22.groupby(year_name).head(top_x_word_per_year)
    
    #take distinct words and top 
    dist_vars = xxd33.variable.unique().tolist()
    
    xxd44 = xxd22[xxd22['variable'].isin(dist_vars)]
    
    #find proportions
    merge1 = pd.merge(xxd44, count_pos1, how='left', on=year_name)
    
    merge1['word_perc'] = merge1.apply(lambda x: (100*x['value']/x[pos_column]) if x[pos_column] > 0 else 0, axis=1)
    
    writer = pd.ExcelWriter(genre+'_'+pos+'_stream_me.xlsx', engine='xlsxwriter');
    merge1.to_excel(writer, sheet_name= 'stream_me');
    writer.save();
    
    return merge1;

#%%
#Word clouds
def wordcloud(freq, year_name, year, pos, max_words, genre):
    
    freq1 = freq[freq[year_name] == year].reset_index(drop=True)
    freq2 = pd.DataFrame(freq1['words'][0]).astype(str)
    freq2 = freq2.rename(index=str, columns={0: 'word', 1: 'count'})
    
    d = {}
    for a, x in freq2.values:
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
    
    plt.savefig('WC_'+genre+'_'+pos+'_'+str(year), bbox_inches='tight')
    
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
    sns.heatmap(data=heatmap3)
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
    for k in frequencies[year_name]:
        wordcloud(frequencies, year_name, k, pos, max_words, genre)
    
    return frequencies, common_words, tabularized; 

#%%
#Wordclouds 

frequencies, common_words, tabularized = run_per_genre(hip_hop, 'Nouns', 'noun_count' ,'hip_hop', 'Year', 100, 5, 10, 20);
#frequencies, common_words, tabularized = run_per_genre(hip_hop, 'Verbs', 'verb_count' ,'hip_hop', 'Year', 100, 5, 10, 20);
#frequencies, common_words, tabularized = run_per_genre(hip_hop, 'Verbs', 'verb_count' ,'hip_hop', 'Year', 100, 5, 10, 20);

        
    
  
    