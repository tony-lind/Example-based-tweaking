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
import matplotlib
#from numpy import genfromtxt
#import matplotlib.pyplot as plt
#import sklearn as sk
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from rf_distance_measures import forest_class_distance
 
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

ex = [[3,5,4,2]] #classified as irisversicolor
wish_class = "irisvirginica" 

#sim_cnt, sim_X, sim_y = forest_class_distance(clf, ex, wish_class, X_train, y_train)

    


#LIME kernel
#Euclidian distance








