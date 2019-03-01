# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:21:41 2019

@author: tony
"""
import numpy as np


def cost_func(a, b):
    return np.linalg.norm(a - b)


def neighbour_tweaking(clf, ex, wish_class, X_train, y_train, classified_as_wish=False):
    temp_best_cost = 100000
    temp_index = -1
    for i in range(len(y_train) - 1):
        correct_class = (y_train[i] == wish_class)
        if correct_class and classified_as_wish:
            t_pred_c = clf.predict([X_train[i]])
            if t_pred_c[0] == wish_class:
                # print("Found example of actual wished class which is classified as such with freq: ", freq)
                temp_cost = cost_func(ex, X_train[i])
                if (temp_cost < temp_best_cost):
                    temp_best_cost = temp_cost
                    temp_index = i 
        elif correct_class:
            # print("Found example of actual wished class with freq:", freq)
            temp_cost = cost_func(ex, X_train[i])
            if (temp_cost < temp_best_cost):
                temp_best_cost = temp_cost
                temp_index = i
    if temp_index != -1:
        return X_train[temp_index]
    else:    
        return ex
            
