# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:31:27 2021

@author: Gourav Beura
"""

#Step 1: Import important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Import dataset
df = pd.read_csv('homeprices.csv')
print(df)

dummies= pd.get_dummies(df.town)
print(dummies)

#Step 3: Concatinate the two dataframes
merged_df = pd.concat([df,dummies],axis='columns')
print(merged_df)

'''
We drop 1 column of dummy data to avoid dummy variable trap
'''
final_df = merged_df.drop(['town','west windsor'],axis='columns')
print(final_df)

#Step 4: Train the model
X = final_df.drop('price',axis='columns')
Y = final_df.price

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4, random_state=0)
reg = LinearRegression()
reg.fit(x_train,y_train)

#Step 5: Predict
predict_result = reg.predict(x_test)
print(predict_result)


#Step 6: Measure Accuracy
acc =  r2_score(y_test,predict_result)
print(acc)

#Step 7: Plot the data
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.scatter(y_test,predict_result)

pred_y_df = pd.DataFrame({'Actual Value':y_test,'Predicted Value':predict_result, 'Difference':y_test-predict_result})
print(pred_y_df)