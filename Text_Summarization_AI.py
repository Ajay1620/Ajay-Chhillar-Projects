# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:54:37 2018

@author: AJAY
"""

import re
import bs4
import urllib.request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence').read()
soup = bs4.BeautifulSoup(scraped_data,'lxml')
text = " "
for paragraph in soup.find_all('p'):
    text +=paragraph.text
    

# Removing the square Brackets and extra spaces

text = re.sub(r'\[[0-9]*\]', ' ',text)
text = re.sub(r'\s+', ' ',text)

# Removing special Chracters and digits
formated_article_text = re.sub(r'[^a-zA-Z]', ' ',text)
formated_article_text = re.sub(r'\s+', ' ',formated_article_text)

sentence_list = sent_tokenize(text)

stop_words = stopwords.words('english')
word_frequencies = {}
for word in word_tokenize(formated_article_text):
    if word not in stop_words:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
            
maximum_frequency = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    
sentence_score = {}
for sent in sentence_list:
    for word in word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_score.keys():
                    sentence_score[sent] = word_frequencies[word]
                else:
                    sentence_score[sent] += word_frequencies[word]

import heapq
summary_sentence = heapq.nlargest(7,sentence_score, key = sentence_score.get)

summary = ' '.join(summary_sentence)
print(summary)

