# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:08:02 2019



@author: Tony Lindgren, tony@dsv.su.se
"""
from collections import Counter
#import pandas as pd
import numpy as np
####
# Method that returns an three arrays, one, containing the number of times the an
# example cooccurs in leaf with ex, two the array of example and three the classes of the examples. 
# For which the following conditions holds true:
# examples share leaf node with example in count number of leafs 
# and is of class whish_class. 
# If classified_as_wish is False (default) is sufficent that the example 
# is of wish_class otherwise it must also be classified as such
def forest_class_distance(clf, ex, wish_class, X_train, y_train, classified_as_wish=False):
    # calculate leaf_id_matrix
    leaf_id_mat = clf.apply(X_train)
    # shape of mat
    (rows, cols) = leaf_id_mat.shape 
    # calulate leaf_example_array
    leaf_ex_arr = clf.apply(ex)   
    # for each tree for each training example check if they share leaf node
    cnt = Counter() 
    for col in range(cols - 1):      # trees
        for row in range(rows - 1):  # example id:s
            if leaf_id_mat[row, col] == leaf_ex_arr[0, col]:
                print("found match at: ", row) 
                cnt[row] += 1
    sorted_cnt = cnt.most_common()
    # filter result on wish_class
    sub_X = np.empty((0, np.size(X_train, 1)))  # define X matrix 
    sub_y = np.empty((0, 1))                    # define y array
    for ex, freq in sorted_cnt:
        correct_class = y_train[ex] == wish_class
        if correct_class and classified_as_wish:
            t_pred_c = clf.predict([X_train[ex]])
            if t_pred_c[0] == wish_class:
                print("Found example of actual wished class which is classified as such with freq: ", freq)
                sub_X = np.vstack([sub_X, X_train[ex]]) 
                sub_y = np.vstack([sub_y, y_train[ex]])   
                #add sorted cnt here
        elif correct_class:
                print("Found example of actual wished class with freq:", freq)
                sub_X = np.vstack([sub_X, X_train[ex]]) 
                sub_y = np.vstack([sub_y, y_train[ex]])
                #add sorted cnt here
    return sorted_cnt, sub_X, sub_y

####
# Method that returns a matrix of the normalised euclidian distance for given exeample (ex) 
# other (training) examples and example_id. 
# For which the following conditions holds true:
# if an feature is numeric then the normalised euclidian distance is used 
# if it is categorical the distance is 0 if same value 1 otherwise.
# If classified_as_wish is False (default) is sufficent that the example 
# is of wish_class otherwise it must also be classified as such
#def euclidian_distance(clf, ex, wish_class, X_train, y_train, classified_as_wish=False):
    