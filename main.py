# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 09:55:52 2019

@author: LENOVO
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from nltk.tokenize import word_tokenize

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['tidy tweet'] = train_df['tweet']
test_df['tidy tweet'] = test_df['tweet']
def preprocess(input,index):
    count=0
    for tweet in input['tweet']:
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub(r'[^\w]', ' ', tweet)
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        #tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)     
        input.iloc[count,index] = tweet
        count = count+1;

preprocess(train_df,3) 
preprocess(test_df,2)  

train_df['tidy tweet'] = train_df['tidy tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
test_df['tidy tweet'] = test_df['tidy tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#Vectorize
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=10)
train_vect = vect.fit_transform(train_df['tidy tweet'])
train_vect = train_vect.toarray()


test_vect = vect.fit_transform(test_df['tidy tweet'])
test_vect = test_vect.toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(
        train_vect, 
        train_df['label'],
        train_size=0.80, 
        random_state=10)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)

from sklearn.metrics import f1_score,accuracy_score
print("score for LR: ", accuracy_score(pred,y_test))

#SVM
from sklearn.svm import SVC
svm =  SVC(kernel='linear')
svm.fit(x_train, y_train)
pred = svm.predict(x_test)
print("score for SVM: ", accuracy_score(pred,y_test))

#RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
print("score for RF: ", accuracy_score(pred,y_test))

#xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
print("score for xgb: ", accuracy_score(pred,y_test))

#For doing the prediction on the test data
lr.fit(train_vect,train_df['label'])
pred = lr.predict(test_vect)
final_result = pd.DataFrame({'id':test_df['id'],'label':pred})
final_result.to_csv('output.csv',index=False)
