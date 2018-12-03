# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 06:50:11 2018

@author: AJAY
"""

import numpy as np
import pandas as pd
import nltk
import re
import bs4
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import requests

from gensim.models import Word2Vec
source = requests.get('https://en.wikipedia.org/wiki/Global_warming')
soup = bs4.BeautifulSoup(source.text,'lxml')
text  = " "

# Fetching the Data
for paragraph in soup.find_all('p'):
    text +=paragraph.text

# Pre-Processing the Data
text = text.lower()
text = re.sub(r'\s+'," ",text)
text = re.sub(r'\d+'," ",text)

# Preparing the Dataset
sentences = sent_tokenize(text)

words = [word_tokenize(sentence) for sentence in sentences]

for i in range(len(words)):
    words[i] = [word for word in words[i] if word not in stopwords.words('english')
                +['[',']','(',')','%']]
    
#Training the word2vec model

model = Word2Vec(words,min_count = 1)
words = model.wv.vocab

# Finding the wordsVec
vector = model.wv['global']

# Most similar word

similar = model.wv.most_similar('global')



 
