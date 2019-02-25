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
#from numpy import genfromtxt
#import matplotlib.pyplot as plt
#import sklearn as sk
#from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from rf_distance_measures import forest_class_distance
from featureTweakPy import feature_tweaking
from cost import cost_func 

## Datasets
# ida2016 
# dataset = pd.read_csv("data/ida_2016_challenge_update/ida_2016_training_set_update.csv")
# Iris
#df = pd.read_csv("C:/Users/tony/CloudStation/dsv/programming/python/Pythonstart/data/iris.csv")
df = pd.read_csv("D:/programmering/python/actionable-features/data/iris.csv")
# Turn classes into integers
df['class-class_names'] = pd.factorize(df['class-class_names'])[0]
iris_values = df.values

# Split into X and y
X_iris = iris_values[:,0:4]
y_iris = iris_values[:,4]

#Split into train and test set
test_size = 0.20
seed = 42
X_iris_train, X_iris_test, y_iris_train, y_iris_test = model_selection.train_test_split(X_iris, y_iris, test_size=test_size, random_state=seed)

# Build RF
forest_size = 10
clf = RandomForestClassifier(n_estimators=forest_size, criterion="entropy")
clf = clf.fit(X_iris_train, y_iris_train)
#clf.score(X_test, y_test) 

missed = 0
tot_rf_cost = 0
tot_ft_cost = 0
# Parameters for feature tweaking
epsilon = 0.1
y_labels = [0, 1, 2]

x = X_iris_test[0]
y = y_iris_test[0]
wish_class = 0
sim_cnt, sim_X, sim_y = forest_class_distance(clf, [x], wish_class, X_iris_train, y_iris_train)
x_new = feature_tweaking(clf, x, y_labels, wish_class, epsilon, cost_func)
"""
for i in range(len(X_test)):
    x = X_test[i]
    y = y_test[i]    
    wish_class = random.randint(0, len(y_labels)-1)
# RF tweaking    
    sim_cnt, sim_X, sim_y = forest_class_distance(clf, [x], wish_class, X_train, y_train)
    if len(sim_X) > 0:
        val_1 = cost_func(sim_X[0], x)
        tot_rf_cost = tot_rf_cost + val_1
    else:
        missed += 1    
# FT 
    x_new = feature_tweaking(clf, x, y_labels, wish_class, epsilon, cost_func)
    val_2 = cost_func(x_new, x)
    tot_ft_cost = tot_ft_cost + val_2
"""
    








