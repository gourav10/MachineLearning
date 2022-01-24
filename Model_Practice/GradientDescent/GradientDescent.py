# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:51:58 2021

@author: Gourav Beura

"""
import numpy as np
import math 
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    
    learning_rate = 0.08
    prev_cost=0
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        
        md = -(2/n)*sum(x*(y-y_predicted))
        mb = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - (learning_rate*md)
        b_curr = b_curr - (learning_rate*mb)
        if math.isclose(prev_cost,cost,rel_tol=1e-20): break
        print("m:{}, b:{}, cost:{}, iteration:{}".format(m_curr,b_curr,cost,i))
        prev_cost = cost

    
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x, y)

