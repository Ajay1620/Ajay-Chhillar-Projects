# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:26:38 2018

@author: AJAY
"""

import re 
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import bs4
import requests
import heapq

source = requests.get('https://en.wikipedia.org/wiki/Global_warming')
soup = bs4.BeautifulSoup(source.text,'lxml')
text = " "
for paragraph in soup.find_all('p'):
    text +=paragraph.text

# Preprocessing the data
 text = re.sub(r'\s+',' ',text)
 text = re.sub(r'\[[0-9]*\]',' ', text)
 clean_text = text.lower() 
 clean_text = re.sub(r'\W', ' ',clean_text)
 clean_text = re.sub(r'\d',' ', clean_text)
 clean_text = re.sub(r'\s+', ' ', clean_text)
 
 # Tokenize the sentences
sentences = sent_tokenize(text)

stop_words = stopwords.words('english')

# Word Count

word2count = {}
for word in word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

# Converting counts to weights

max_count = max(word2count.values())
for key in word2count.keys():
    word2count[key] = word2count[key]/max_count
    
# Product sentence scores    
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 35:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]
                    
# Gettings best 5 lines             
best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)

print('---------------------------------------------------------')
for sentence in best_sentences:
    print(sentence)


        









