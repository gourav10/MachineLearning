# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 00:57:52 2021

@author: Gourav Beura
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model,model_selection


df = pd.read_csv('insurance_data.csv')
print(df.head(5))


#Visualize the dataset
plt.xlabel('Age')
plt.ylabel('insurance taken')
plt.scatter(df.age,df.bought_insurance,marker='+',color='red')


#Split data into training and test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[['age']],df[['bought_insurance']],train_size=0.9,random_state=0)
print(X_test)

print(y_test)
#Train Model
reg = linear_model.LogisticRegression()
reg.fit(X_train,y_train)


print(reg.predict(X_test))

print(reg.score(X_test,y_test))