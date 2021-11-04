# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:56:34 2021

@author: furka
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import sem
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot


data=pd.read_csv("C://Users//furka//OneDrive//Masaüstü//telefon_fiyat_değişimi.csv")
print(data)



#Sınıf Sayısı Belirleme
label_encoder=LabelEncoder().fit(data.price_range)
labels=label_encoder.transform(data.price_range)
classes=list(label_encoder.classes_)

x=data.drop(["price_range"],axis=1)
y=labels

#verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
print(x)



def evaluate_model(x, y, repeats):
	# prepare the cross-validation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
	# create model
	model = LogisticRegression()
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores


x, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)

repeats = range(1,16)
results = list()
for r in repeats:
	
	scores = evaluate_model(x, y, r)
	
	print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
	
	results.append(scores)

pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
pyplot.show()










