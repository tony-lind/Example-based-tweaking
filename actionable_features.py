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
from cost import cost_func, neighbour_tweaking, to_closest_int

datasets = [("iris", "C:/jobb/programmering/PythonDev/data/numerical_data/iris.csv", "yes"),
            #("glass", "d:/programmering/Python/actionable-features/data/glass.csv","yes") #,
            #("arrhythmia", "d:/programmering/Python/actionable-features/data/arrhythmia_replaced_Nan_w_0.csv","yes")
            #("b_c_wisconsin", "d:/programmering/Python/actionable-features/data/b_c_wisconsin.csv","yes")
            #("bupa", "d:/programmering/Python/actionable-features/data/bupa.csv","yes")
            #("c_h_disease", "d:/programmering/Python/actionable-features/data/c_h_disease.csv","yes")
            #("climate model_failures", "d:/programmering/Python/actionable-features/data/climate_model_failures.csv","yes")
            #("covtype", "d:/programmering/Python/actionable-features/data/covtype.csv","yes")
            #("default_of_credit_card_clients", "d:/programmering/Python/actionable-features/data/default_of_credit_card_clients.csv","yes")
            #("forest_types", "d:/programmering/Python/actionable-features/data/forest_types.csv","yes") #error in feature tweaking..
            #("haberman", "d:/programmering/Python/actionable-features/data/haberman.csv","yes") #error in feature tweaking..
            #("image_seg", "c:/jobb/programmering/PythonDev/actionable-features/data/image_seg.csv","yes") 
            #("ionosphere", "c:/jobb/programmering/PythonDev/actionable-features/data/ionosphere.csv","yes")
            #("ionosphere", "c:/jobb/programmering/PythonDev/actionable-features/data/ionosphere.csv","yes")
            #("magic4", "C:/jobb/programmering/PythonDev/actionable-features/data/magic04.csv","yes")
            #("sensorless_drive_diagnosis", "C:/jobb/programmering/PythonDev/actionable-features/data/sensorless_drive_diagnosis.csv","yes")
            #("shuttle", "C:/jobb/programmering/PythonDev/data/categorical_data/shuttle.csv","yes")
            #("connect_4", "C:/jobb/programmering/PythonDev/data/categorical_data/connect_4.csv","yes")
            #("tictactoe", "C:/jobb/programmering/PythonDev/data/categorical_data/tictactoe.csv","yes")
            #("zoo", "C:/jobb/programmering/PythonDev/data/categorical_data/zoo.csv","yes")
            #("king_rook_vs_king_pawn", "C:/jobb/programmering/PythonDev/data/categorical_data/king_rook_vs_king_pawn.csv","yes")       
            #("car", "C:/jobb/programmering/PythonDev/data/categorical_data/car.csv","yes")   
            #("1625Data", "C:/jobb/programmering/PythonDev/data/categorical_data/hiv/1625Data.csv","yes")  
            #("746Data", "C:/jobb/programmering/PythonDev/data/categorical_data/hiv/746Data.csv","yes")  
            #("impens", "C:/jobb/programmering/PythonDev/data/categorical_data/hiv/impens.csv","yes") 
            #("schilling", "C:/jobb/programmering/PythonDev/data/categorical_data/hiv/schilling.csv","yes")
            #("nursery", "C:/jobb/programmering/PythonDev/data/categorical_data/nursery.csv","yes")  
            #("titanic", "C:/jobb/programmering/PythonDev/data/categorical_data/titanic.csv","yes")  
            
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
#df = pd.read_csv("C:/jobb/programmering/PythonDev/data/categorical_data/shuttle.csv")
    df = pd.read_csv(d_s)
# Turn classes into integers
    if(trans == "yes"):
        df['class-_'] = pd.factorize(df['class-_'])[0]
        for i in range(0, len(df.columns) - 1):   #class always last column
            t_name = df.iloc[:,i].name
            if('categoric-' in t_name):
                df[t_name] = pd.factorize(df[t_name])[0]     
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
    print("Build model of size 10")
    our_models.append(("10", clf_10))
    
    clf_50 = RandomForestClassifier(n_estimators=forest_size[1], criterion="entropy", n_jobs=-1)
    clf_50 = clf_50.fit(X_train, y_train)
    print("Build model of size 50")
    our_models.append(("50", clf_50))
    
    clf_100 = RandomForestClassifier(n_estimators=forest_size[2], criterion="entropy", n_jobs=-1)
    clf_100 = clf_100.fit(X_train, y_train)
    print("Build model of size 100")
    our_models.append(("100", clf_100))
    
    clf_250 = RandomForestClassifier(n_estimators=forest_size[0], criterion="entropy", n_jobs=-1)
    clf_250 = clf_250.fit(X_train, y_train)
    print("Build model of size 250")
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
    print("Starting evaluation of tweaking methods")
    for (m_size, our_model) in our_models:       
        # Reset performance metric values  
        missed_rft = 0
        tot_rft_cost = 0
        missed_nt = 0
        tot_nt_cost = 0
        missed_ft = 0
        tot_ft_cost = 0
        tot_ft_int_cost = 0
        val_1 = 0
        val_2 = 0
        val_3 = 0
        val_4 = 0
        print("evaluation of tweaking starts")    
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
            #print("starting random forest tweaking") 
            sim_cnt, sim_X, sim_y = random_forest_tweaking(our_model, [x], wish_class, X_train, y_train)
            if len(sim_X) > 0:
                #print("RFT: ", sim_X[0])
                val_1 = cost_func(sim_X[0], x)
                tot_rft_cost = tot_rft_cost + val_1
            else:
                missed_rft += 1    
              
            # Neighbour tweaking
            #print("starting neighbour tweaking")
            ex_neighbour = neighbour_tweaking(our_model, [x], wish_class, X_train, y_train)
            if(np.equal(ex_neighbour, x).all()):
                missed_nt += 1
            else:
                #print("NT: ", ex_neighbour)
                val_2 = cost_func(ex_neighbour, x)
                tot_nt_cost = tot_nt_cost + val_2
                 
            # Feature tweaking 
            #print("starting feature tweaking")
            x_new_ft = feature_tweaking(our_model, x, y_labels, wish_class, epsilon, cost_func)
            if(np.equal(x_new_ft, x).all()):
                missed_ft += 1
            else:
                #print("FT: ", x_new_ft)
                int_x_ft = to_closest_int(x_new_ft) 
                #print("FT: ", int_x_ft)               
                val_3 = cost_func(x_new_ft, x)
                val_4 = cost_func(int_x_ft, x)
                tot_ft_cost = tot_ft_cost + val_3
                tot_ft_int_cost = tot_ft_int_cost + val_4
                 
        no_ex = len(X_test)    
        if (no_ex - missed_rft) != 0:
            norm_val_1 = tot_rft_cost / (no_ex - missed_rft)
        else:
            norm_val_1 = 0    
        if (no_ex - missed_nt) != 0:
            norm_val_2 = tot_nt_cost / (no_ex - missed_nt)
        else:
            norm_val_2 = 0     
        if (no_ex - missed_ft) != 0:
            norm_val_3 = tot_ft_cost / (no_ex - missed_ft)
            norm_val_4 = tot_ft_int_cost / (no_ex - missed_ft)
        else:
            norm_val_3 = 0    
            norm_val_4 = 0
        results.append(("random_forest_tweaking", no_ex, m_size, d_name, missed_rft, tot_rft_cost, norm_val_1, 0, 0))   
        results.append(("neighbour_tweaking", no_ex, m_size, d_name, missed_nt, tot_nt_cost, norm_val_2, 0, 0))  
        results.append(("feature_tweaking", no_ex, m_size, d_name, missed_ft, tot_ft_cost, norm_val_3, tot_ft_int_cost, norm_val_4))   
    
for (method_name, no_ex, m_size, data_set_name, missed_vals, distance_val, norm_distance_val, int_d_val, norm_d_int_val) in results:
    print("Method: ", method_name)
    print("No examples: ", no_ex)
    print("Model size: ", m_size)
    print("Data set: ", data_set_name)
    print("Missed cases: ", missed_vals)
    print("Total Distance cost: ", distance_val)
    print("Normalized Distance cost: ", norm_distance_val)
    print("Total Inteteger Distance cost: ", int_d_val)
    print("Normalized Integer Distance cost: ", norm_d_int_val)
    
with open("c:/jobb/programmering/PythonDev/actionable-features/results", 'w') as file_handler:
    for item in results:
        file_handler.write("{}\n".format(item))







