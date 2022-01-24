# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:53:25 2021

@author: Skyfire
"""
import math
import pandas as pd
from sklearn import linear_model

dataframe = pd.read_csv('homeprices.csv')
print(dataframe) #There is null data

'''
remove the null data from dataset by filling null value with median(safe assumption)

'''
#Step: Data Preprocessing and cleaning
median_bedrooms = math.floor(dataframe.bedrooms.median())
print(math.floor(dataframe.bedrooms.median()))
print(dataframe.bedrooms.fillna(median_bedrooms))
dataframe.bedrooms = dataframe.bedrooms.fillna(median_bedrooms)
print(dataframe)

#Step 3: Linear Regression
reg = linear_model.LinearRegression()
reg.fit(dataframe[['area','bedrooms','age']],dataframe.price)

print("Coefficients: {}, Intercept: {}".format(reg.coef_,reg.intercept_))

print("Predicted Price of House with Area: {}, Bedrooms: {}, Age: {} is ${}"
      .format(3000,3,12,reg.predict([[3000,3,40]])))
