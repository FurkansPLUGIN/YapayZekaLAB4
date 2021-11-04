# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:17:56 2021

@author: furka
"""


#1.soru
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

#eğitim ve test verilerinin hazırlanması


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.2)

#çıktı değerlerini katogorileştirme  #binary formatına çevrildi
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#ysa modelinin oluşturulması

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(16,input_dim=20,activation="relu")) #girdi katmanı

model.add(Dense(12,activation="relu")) #ara katman
model.add(Dense(8,activation="relu"))
model.add(Dense(6,activation="relu"))


model.add(Dense(4,activation="softmax")) #çıktı katmanı 4 tane sınıf olduğu için 4 nöron
model.summary()

#modelin derlenmesi
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

#modelin derlenmesi
model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=50)

#gerekli değerlerin yazdırılması
print("Ortalama eğitim kaybı: ",np.mean(model.history.history["loss"]))

print("Ortalama eğitim başarımı: ",np.mean(model.history.history["accuracy"]))

print("Ortalama doğrulama kaybı: ",np.mean(model.history.history["val_loss"]))

print("Ortalama doğrulama başarımı: ",np.mean(model.history.history["val_accuracy"]))

#eğitim ve doğrulama başarımlarının gösterilmesi

import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Başarım")
plt.xlabel("epok")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show()

















