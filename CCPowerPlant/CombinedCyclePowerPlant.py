# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 07:17:10 2021

@author: Gourav Beura

"""


#Step 1: Import Libraries
import pandas as pd
from sklearn import linear_model,model_selection,metrics
import matplotlib.pyplot as plt

#Step 2: Import data set
xls = pd.ExcelFile('CCPP\Folds5x2_pp.xlsx')
df = pd.read_excel(xls,'Sheet1')
print("Test Data:")
print(df.head(5))

#Step 3: Define X and Y i.e. independent and dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Step 4: Split dataset in training and test set
x_train,x_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.3, random_state=0)

#Step 5: Train the model
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)

#Step 6: Predict the model
y_predict = reg.predict(x_test)
print("Predicted Result:")
print(y_predict)

#Step 7: Evaluate the model
acc = metrics.r2_score(y_test,y_predict)
print(acc)

#Step 8: Plot the results
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.scatter(y_test,y_predict)

#Step 9: Print predicted values
pred_y_df = pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_predict, 'Difference':y_test-y_predict})
print(pred_y_df.head(10))