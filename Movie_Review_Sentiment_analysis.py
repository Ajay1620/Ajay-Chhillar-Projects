# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:45:19 2018

@author: AJAY
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
path = "C:\\Users\\AJAY\\NLP\\Dats sets\\User_Reviews\\User_Reviews\\User_movie_review.csv"
data = pd.read_csv(path)
data.head()
stemmer = PorterStemmer()

def tokenize(text):
    text = stemmer.stem(text)
    text = re.sub(r'\W+|\d+_', ' ',text)
    tokens = word_tokenize(text)
    return tokens

from sklearn.feature_extraction.text import CountVectorizer

countvec1 = CountVectorizer(min_df = 5, tokenizer = tokenize, stop_words = stopwords.words('english'))

dataset = pd.DataFrame(countvec1.fit_transform(data['class']).toarray(), 
                       columns = countvec1.get_feature_names(), index = None)

dataset['class'] = data['class']

dataset.head(10)

#Dividing training and testing sets

df_train = dataset[:1900]
df_test = dataset[1900:]

# Building Naive Bayes Model

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
X_train = df_train.drop(['class'], axis = 1)

# Fitting model to our data
clf.fit(X_train, df_train['class'])

# To check the Accuracy of model
X_test = df_test.drop(['class'], axis = 1)
clf.score(X_test, df_test['class'])

# Predication
pred_sentiment = clf.predict(df_test.drop('class', axis = 1))
print(pred_sentiment)
