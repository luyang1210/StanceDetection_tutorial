# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:15:16 2020

"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import text
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from os import path
import pandas as pd
import logging
import matplotlib.pyplot as plt
from io import StringIO



"""Takes in dataframe, extracts column"""
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        #print(data_dict[self.key])
        for t in data_dict[self.key]:
            #print(t)
            return data_dict[self.key]



def Supervisedmodel( training_file,testing_file):      
    training_data = pd.read_csv(training_file, sep='\t', encoding='latin1', low_memory=False)    
    print("training")
    print(training_data.shape)    
    
    print("----------------------------------")
    testing_data = pd.read_csv(testing_file,  sep='\t', encoding='latin1',low_memory=False)
    print("Testing")
    print(testing_data.shape)
   
    #-----------training feature-----------------------
    my_stopword_list= text.ENGLISH_STOP_WORDS
    Y_train = np.asarray([stance for stance in training_data['label']]) 
    Y_eval = np.asarray([stance for stance in testing_data['label']])
    my_stopword_list
    # build the feature matrices
    
    tweet_hashtag= Pipeline([
                    ('selector', ItemSelector(key='hashtags')),
                    ('count', CountVectorizer(analyzer='word',binary=False,ngram_range=(1,1))),
                ])
   
    tweet_word= Pipeline([
                    ('selector', ItemSelector(key='tweet')),
                    ('count', CountVectorizer(analyzer='word',binary=True,ngram_range=(1,2))),
                ])
    tweet_char= Pipeline([
                   ('selector', ItemSelector(key='tweet')),
                    ('count', CountVectorizer(analyzer='char',binary=True,ngram_range=(1,3))),
                ])
   
 
    
    
    ppl = Pipeline([
      
         ('feats', FeatureUnion([
 
                  ("tweet_word", tweet_word),

                     ('tweet_char',tweet_char),

                   ('tweet_hashtag',tweet_hashtag),

      ]))
  
       ,('clf', SVC(kernel='linear',class_weight='balanced') )
    ])    
    
        
    model = ppl.fit(training_data, Y_train)
    model.fit(training_data, Y_train)
    print(model.classes_)

    y_test = model.predict(testing_data)
    print(classification_report(Y_eval, y_test))
 
           
    
if __name__ == '__main__':
    training_file = ""
    testing_file = ""
    Supervisedmodel(training_file,testing_file)