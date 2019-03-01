# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:07:49 2019

Code for testing different distance measures using actionable features 

@author: Tony Lindgren, tony@dsv.su.se
"""
#import scipy
#
import numpy as np
import pandas as pd
#import matplotlib
import random
#from numpy import genfromtxt
#import matplotlib.pyplot as plt
#import sklearn as sk
#from sklearn import datasets

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from rf_distance_measures import random_forest_tweaking
from featureTweakPy import feature_tweaking
from cost import cost_func, neighbour_tweaking

datasets = [#("iris", "d:/programmering/Python/actionable-features/data/iris.csv", "yes"),
            #("glass", "d:/programmering/Python/actionable-features/data/glass.csv","yes") #,
            ("arrhythmia", "d:/programmering/Python/actionable-features/data/arrhythmia_replaced_Nan_w_0.csv","yes")
            #("magic4", "C:/jobb/programmering/PythonDev/actionable-features/data/magic04.csv","yes")
            ]

results = []
forest_size = [10, 50, 100, 250]
for (d_name, d_s, trans) in datasets:
## Datasets
# ida2016 
# dataset = pd.read_csv("data/ida_2016_challenge_update/ida_2016_training_set_update.csv")
# Iris
#df = pd.read_csv("C:/Users/tony/CloudStation/dsv/programming/python/Pythonstart/data/iris.csv")
#df = pd.read_csv("C:/jobb/programmering/PythonDev/actionable-features/data/iris.csv")
#df = pd.read_csv("C:/jobb/programmering/PythonDev/actionable-features/data/glass.csv")
    df = pd.read_csv(d_s)
# Turn classes into integers
    if(trans == "yes"):
        df['class-_'] = pd.factorize(df['class-_'])[0]
    values = df.values
# Split into X and y - note that class label MUST be in the LAST column
    X = values[:,0:len(df.columns) - 1] 
    y = values[:,len(df.columns) - 1]

# Split into train and test set
    test_size = 0.20
    seed = 42
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)

# Build RF - Vary the forest size? [10, 50, 100, 250]
    our_models = []
    clf_10 = RandomForestClassifier(n_estimators=forest_size[0], criterion="entropy", n_jobs=-1)
    clf_10 = clf_10.fit(X_train, y_train)
    our_models.append(("10", clf_10))
    clf_50 = RandomForestClassifier(n_estimators=forest_size[1], criterion="entropy", n_jobs=-1)
    clf_50 = clf_50.fit(X_train, y_train)
    our_models.append(("50", clf_50))
    clf_100 = RandomForestClassifier(n_estimators=forest_size[2], criterion="entropy", n_jobs=-1)
    clf_100 = clf_100.fit(X_train, y_train)
    our_models.append(("100", clf_100))
    clf_250 = RandomForestClassifier(n_estimators=forest_size[3], criterion="entropy", n_jobs=-1)
    clf_250 = clf_250.fit(X_train, y_train)
    our_models.append(("250", clf_250))
    
# Parameters for feature tweaking
    epsilon = 0.5 # as this gives the highest coverage of 77.4 according to their paper
    y_labels = np.unique(y)

#x = X_iris_test[0]
#print(x)
#y = y_iris_test[0]
#wish_class = 0
#print(y)
#sim_cnt, sim_X, sim_y = random_forest_tweaking(clf, [x], wish_class, X_iris_train, y_iris_train)
#x_new = feature_tweaking(clf, x, y_iris_labels, wish_class, epsilon, cost_func)
#print(x_new)

    for (m_size, our_model) in  our_models:       
        # Reset performance metric values  
        missed_rft = 0
        rft_cost = 0
        missed_nt = 0
        tot_nt_cost = 0
        missed_ft = 0
        tot_ft_cost = 0
            
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]    
            wish_class = random.randint(0, len(y_labels)-1)
            if(y == wish_class):
                if(wish_class == 0):
                    wish_class = 1
                else:
                    wish_class -= 1
            #print("x: ", x)
            #print("y: ", y)
            #print("wish_class: ", wish_class)   
        
            # Random forest tweaking            
            sim_cnt, sim_X, sim_y = random_forest_tweaking(our_model, [x], wish_class, X_train, y_train)
            if len(sim_X) > 0:
                val_1 = cost_func(sim_X[0], x)
                rft_cost = rft_cost + val_1
            else:
                missed_rft += 1    
              
            # Neighbour tweaking
            ex_neighbour = neighbour_tweaking(our_model, [x], wish_class, X_train, y_train)
            if(np.equal(ex_neighbour, x).all()):
                missed_nt += 1
            else:
                val_2 = cost_func(ex_neighbour, x)
                tot_nt_cost = tot_nt_cost + val_2
                 
            # Feature tweaking 
            x_new_ft = feature_tweaking(our_model, x, y_labels, wish_class, epsilon, cost_func)
            if(np.equal(x_new_ft, x).all()):
                missed_ft += 1
            else:
                val_3 = cost_func(x_new_ft, x)
                tot_ft_cost = tot_ft_cost + val_3    
        no_ex = len(X_test)    
        norm_val_1 = val_1 / (no_ex - missed_rft)
        norm_val_2 = val_2 / (no_ex - missed_nt)
        norm_val_3 = val_3 / (no_ex - missed_ft)
        results.append(("random_forest_tweaking", no_ex, m_size, d_name, missed_rft, val_1, norm_val_1))   
        results.append(("neighbour_tweaking", no_ex, m_size, d_name, missed_nt, val_2, norm_val_2))  
        results.append(("feature_tweaking", no_ex, m_size, d_name, missed_ft, val_3, norm_val_3))   
    
for (method_name, no_ex, m_size, data_set_name, missed_vals, distance_val, norm_distance_val) in results:
    print("Method: ", method_name)
    print("No examples: ", no_ex)
    print("Model size: ", m_size)
    print("Data set: ", data_set_name)
    print("Missed cases: ", missed_vals)
    print("Total Distance cost: ", distance_val)
    print("Normalized Distance cost: ", norm_distance_val)
    
with open("d:/programmering/Python/actionable-features/results", 'w') as file_handler:
    for item in results:
        file_handler.write("{}\n".format(item))







