# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 06:33:43 2018

@author: AJAY
"""

import nltk
from nltk.corpus import PlaintextCorpusReader

doc_dirname_poltics = "C:\\Users\\AJAY\\NLP\\20_newsgroups\\20_newsgroups\\talk.politics.misc"
doc_dirname_comps =  "C:\\Users\\AJAY\\NLP\\20_newsgroups\\20_newsgroups\\comp.os.ms-windows.misc"
poltics_news_corpus = PlaintextCorpusReader(doc_dirname_poltics,'.*')
poltics_news_corpus.fileids()
comp_news_corpus = PlaintextCorpusReader(doc_dirname_comps,'.*')

import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def custom_preprocessor(text):
    text = re.sub(r'\W+|\d+|_', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if not word in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

politics_news_docs = [(custom_preprocessor(poltics_news_corpus.raw(fileid)),'poltics')
                       for fileid in poltics_news_corpus.fileids()]

comp_news_docs = [(custom_preprocessor(comp_news_corpus.raw(fileid)),'comp')
                   for fileid in comp_news_corpus.fileids()]

politics_news_docs[0]
comp_news_docs[0]

documents = politics_news_docs + comp_news_docs

import random
random.seed(50)

random.shuffle(documents)

documents[0]

documents[1]

all_words = []
for t in documents:
    for w in t[0]:
        all_words.append(w)
        
        
all_words_freq = nltk.FreqDist(all_words)
print(all_words_freq.most_common(150))

print(all_words_freq['president'])

word_features = all_words

word_features


def find_features(docs):
    words = set(docs)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        return features
    
featuresets = [(find_features(rev), category) for (rev, category) in documents]
featuresets[1:5]

#Creating Training and Testing Data
train_set = featuresets[:160]
test_set = featuresets[160:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Accuracy is :", nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(25)

