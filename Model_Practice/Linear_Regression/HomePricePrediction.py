# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:13:10 2021

We will use a dummy csv file to extract data about the prices and areas of house.
Train the Linear Regression Model to learn the pattern and then predict data.   

@author: Skyfire
"""

#Step 1: Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('homeprices.csv')
print(df)

plt.xlabel('area(sq. ft.')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price,color='red',marker='+')

#Step 3: 
reg = LinearRegression()
reg.fit(df[['area']],df.price)

print("If Area of land is ${} the predicted Price:{}".format(3300, 
                                           reg.predict( np.array(3300).reshape(-1,1))))


#Step 4: Create Test Data frame
test_df = pd.read_csv('areas.csv')

print(test_df.head(3))

p = reg.predict(test_df)
test_df['prices'] = p
print(test_df)

plt.plot(df.area,reg.predict(df[['area']]),color='blue')
