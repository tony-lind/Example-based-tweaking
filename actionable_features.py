# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:07:49 2019

Code for testing different distance measures using actionable features 

@author: Tony Lindgren, tony@dsv.su.se
"""
#import scipy
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from rf_distance_measures import forest_class_distance
 
##datasets
#ida2016 
#dataset = pd.read_csv("data/ida_2016_challenge_update/ida_2016_training_set_update.csv")
#Iris
#df = pd.read_csv("C:/Users/tony/CloudStation/dsv/programming/python/Pythonstart/data/iris.csv")
df = pd.read_csv("data/iris.csv")
val = df.values
#Split into X and y o 
X = val[:,0:4]
y = val[:,4]

#data preparation
test_size = 0.20
seed = 42
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)


#build RF
forest_size = 10
clf = RandomForestClassifier(n_estimators=forest_size, criterion="entropy")
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)

ex = [[3,5,4,2]] #classified as irisversicolor
wish_class = "irisvirginica" 

freq_cnt, sim_X, sim_y = forest_class_distance(clf, ex, wish_class, X_train, y_train, True)

    


#LIME kernel
#Euclidian distance








