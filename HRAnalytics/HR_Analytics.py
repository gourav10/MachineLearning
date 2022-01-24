# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 02:22:02 2021

@author: Gourav Beura
"""

#Step 1: Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model

#Step 2: Import the dataset
df = pd.read_csv('HR_comma_sep.csv')
print(df.head(10))

#Step 3: Understand the data in dataset using different 
print(df.info())
print(df.describe())

#Create Histograms for better understandnig the relations
df_num = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','salary']]
df_cat = df[['left','Work_accident','promotion_last_5years','Department']]
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()    