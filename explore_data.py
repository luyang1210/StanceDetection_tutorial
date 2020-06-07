# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:57:12 2020

@author: Abeer
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:29:19 2020

@author: Abeer
"""

from os import path
import pandas as pd
import pdb
from IPython.core.debugger import Tracer
import logging
import re
import matplotlib.pyplot as plt
from io import StringIO
import re
import string
from shifterator import relative_shift as rs
from shifterator import symmetric_shift as ss
from nltk.corpus import stopwords



training_file = " "

testing_file = " "


def init():
    global training_data
    global testing_data
    #encoding 'cp1252'
    training_data = pd.read_csv(training_file,  delimiter='\t', encoding='latin1')   
    testing_data = pd.read_csv(testing_file, delimiter='\t', encoding='latin1')


def freq_words(data):
    frequency = {}
    cachedStopWords=stopwords.words('english')
    ht=[ht for ht in data['hashtags']]
    text=" "
    stance_text=""
    for h in ht:
        for h1 in h.split():
            stance_text=stance_text+"\t"+h1
            
    text = ' '.join([word for word in stance_text.split() if word not in cachedStopWords])

    text_string = text.lower()
    match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)
 
    for word in match_pattern:
        count = frequency.get(word,0)
        frequency[word] = count + 1
     
    frequency_list = frequency.keys()
 
    for words in frequency_list:
        print (words, frequency[words])
        
    return frequency


def explore_hashtags(data):
    print("hashtages_training")
    #--- anti hashtags
    results_class=['anti']
    data_anti= data[ data["label"].isin(results_class)] 
    anti_freq=freq_words(data_anti)
    
    #--- pro hashtags
    results_class=['pro']
    data_anti= data[ data["label"].isin(results_class)] 
    pro_freq=freq_words(data_anti)
    
    #The Jensen-Shannon divergence symmetrizes the Kullback-Leibler divergence by measuring 
    #the average divergence of each text from another text representing their average. 
    #The measure is symmetric, meaning there is no order in how the texts are specified.
    #---------------------
    #Display the top_n types as sorted by their absolute contribution to the difference between systems
    jsd_shift = ss.JSDivergenceShift(system_1=pro_freq,
                                 system_2=anti_freq,
                                 base=2)
    jsd_shift.get_shift_graph()

    

if __name__ == '__main__':
         
    logging.info('initialising...')
    init()
    download_more_data = False
    plot=True
    check_tweet_by_ID=False
    explore_htg=True
    
    if explore_htg:
       explore_hashtags(training_data) 
    
    if plot:
        print("------Testing plot------")        
        
        col = ['tweet', 'label']
        
        testing_data= testing_data[col]
        
        plt.style.use('seaborn-darkgrid')
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}	
        plt.rc('font', **font)
        testing_data.head() 
        fig = plt.figure(figsize=(8,6))
        fig_result2=testing_data.groupby('label').tweet.count().plot.bar(ylim=0)
        for p in fig_result2.patches:
             fig_result2.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        
        plt.show()
        
        print("------training plot------") 
        col = ['tweet', 'label']
        training_data= training_data[col]
       
        training_data.head() 
        fig = plt.figure(figsize=(8,6))
        fig_result=training_data.groupby('label').tweet.count().plot.bar(ylim=0)
        for p in fig_result.patches:
             fig_result.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        print("-----------------------------------data pivot--------------------------------------")
            
        plt.show()
        