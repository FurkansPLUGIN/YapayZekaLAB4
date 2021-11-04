# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:29:02 2021

@author: furka
"""

#5.soru

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("C://Users//furka//OneDrive//Masaüstü//diyabet.csv")
print(data)

#"‪C://Users//furka//OneDrive//Masaüstü//diyabet.csv

#Sınıf Sayısı Belirleme
label_encoder=LabelEncoder().fit(data.output)
labels=label_encoder.transform(data.output)
classes=list(label_encoder.classes_)


x=data.drop(["output"],axis=1)
y=labels

#verileri bölme
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#verileri eğitme
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


clf_tree = DecisionTreeClassifier();
clf_reg = LogisticRegression();

clf_tree.fit(x_train, y_train); 
clf_reg.fit(x_train, y_train);

#modellerin test edilmesi

y_score1 = clf_tree.predict_proba(x_test)[:,1]
y_score2 = clf_reg.predict_proba(x_test)[:,1]

#true false durumların incelenmesi
from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)

#auc skor işlemi
print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_score1))
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_score2))


#cizdirme

#oran 1
import matplotlib.pyplot as plt
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#oran2
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()










































