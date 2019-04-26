# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:07:49 2019

Code for testing different distance measures using actionable features 

@author: Tony Lindgren, tony@dsv.su.se
"""
import numpy as np
import pandas as pd
import random
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from rf_distance_measures import random_forest_tweaking
from featureTweakPy import feature_tweaking
from cost import cost_func, neighbour_tweaking, to_closest_int

datasets = [# Numerical data sets
            #("b_c_wisconsin", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/b_c_wisconsin.csv","yes")
            #("bupa", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/bupa.csv","yes")
            #("c_h_disease", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/c_h_disease.csv","yes")
            #("climate_model_failures", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/climate_model_failures.csv","yes")
            #("glass", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/glass.csv","yes")
            #("image_seg", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/image_seg.csv","yes") 
            #("ionosphere", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/ionosphere.csv","yes")
            #("iris", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/iris.csv", "yes")
            #("magic04", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/magic04.csv","yes")
            #("pendigits", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/pendigits.csv","yes")
            #("pima_indians", "c:/jobb/programmering/PythonDev/actionable-features/data/numerical/pima_indians.csv","yes")
            # Categorical data sets               
            #("balance_scale", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/balance_scale.csv","yes")
            #("car", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/car.csv","yes")
            #("HIV_746", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/HIV_746.csv","yes")
            #("HIV_1625", "C:/jobb/programmering/PythonDev/actionable-features/data/categorical/HIV_1625.csv","yes")
            #("impens", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/impens.csv","yes")
            #("KRKP7", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/KRKP7.csv","yes"),          
            #("promoters", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/promoters.csv","yes")
            #("schilling", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/schilling.csv","yes")  
            ("shuttle", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/shuttle.csv","yes")
            #("tic_tac_toe", "c:/jobb/programmering/PythonDev/actionable-features/data/categorical/tictactoe.csv","yes")                               
            #("zoo", "C:/jobb/programmering/PythonDev/actionable-features/data/categorical/zoo.csv","yes")            
            ]

results = []
forest_size = [10, 50, 100, 250]
for (d_name, d_s, trans) in datasets:
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
    print("Starting evaluation of tweaking methods")
    for (m_size, our_model) in our_models:       
        # Reset performance metric values  
        missed_rft = 0
        tot_rft_cost = 0
        missed_nt = 0
        tot_nt_cost = 0
        missed_ft = 0
        tot_ft_cost = 0
        val_1 = 0
        val_2 = 0
        val_3 = 0
       
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
                #print("FT: ", int_x_ft)               
                val_3 = cost_func(x_new_ft, x)               
                tot_ft_cost = tot_ft_cost + val_3               
                 
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
        else:
            norm_val_3 = 0    
        
        results.append(("random_forest_tweaking", no_ex, m_size, d_name, missed_rft, tot_rft_cost, norm_val_1))   
        results.append(("neighbour_tweaking", no_ex, m_size, d_name, missed_nt, tot_nt_cost, norm_val_2))  
        results.append(("feature_tweaking", no_ex, m_size, d_name, missed_ft, tot_ft_cost, norm_val_3))   
    
for (method_name, no_ex, m_size, data_set_name, missed_vals, distance_val, norm_distance_val) in results:
    print("Method: ", method_name)
    print("No examples: ", no_ex)
    print("Model size: ", m_size)
    print("Data set: ", data_set_name)
    print("Missed cases: ", missed_vals)
    print("Total Distance cost: ", distance_val)
    print("Normalized Distance cost: ", norm_distance_val)
    
with open("c:/jobb/programmering/PythonDev/actionable-features/results", 'w') as file_handler:
    for item in results:
        file_handler.write("{}\n".format(item))







