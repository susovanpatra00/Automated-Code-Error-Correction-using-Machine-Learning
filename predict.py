import numpy as np
import pickle
from numpy import random as rand
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import accuracy_score 

def loadData( filename, dictSize = 225 ):
    X, y = load_svmlight_file( filename, multilabel = False, n_features = dictSize, offset = 1 )
    return (X, y)

X,y=loadData("train")
print(X.shape, y.shape)

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# PLEASE BE CAREFUL THAT ERROR CLASS NUMBERS START FROM 1 AND NOT 0. THUS, THE FIFTY ERROR CLASSES ARE
# NUMBERED AS 1 2 ... 50 AND NOT THE USUAL 0 1 ... 49. PLEASE ALSO NOTE THAT ERROR CLASSES 33, 36, 38
# NEVER APPEAR IN THE TRAINING SET NOR WILL THEY EVER APPEAR IN THE SECRET TEST SET (THEY ARE TOO RARE)

# Input Convention
# X: n x d matrix in csr_matrix format containing d-dim (sparse) bag-of-words features for n test data points
# k: the number of compiler error class guesses to be returned for each test data point in ranked order

# Output Convention
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of classes with the i-th row 
# containing k error classes which it thinks are most likely to be the correct error class for the i-th test point.
# Class numbers must be returned in ranked order i.e. the label yPred[i][0] must be the best guess for the error class
# for the i-th data point followed by yPred[i][1] and so on.

# CAUTION: Make sure that you return (yPred below) an n x k numpy nd-array and not a numpy/scipy/sparse matrix
# Thus, the returned matrix will always be a dense matrix. The evaluation code may misbehave and give unexpected
# results if an nd-array is not returned. Please be careful that classes are numbered from 1 to 50 and not 0 to 49.


def findErrorClass( X, k ):
    # Find out how many data points we have
    n = X.shape[0]
    # Load and unpack a dummy model to see an example of how to make prediction
    # The dummy model simply stores the error classes in decreasing order of their popularity
    clf = pickle.load(open('code_corrector2.pickle', 'rb'))
    
    y_trial= clf.predict_proba(X) #to get the probs of all the classes to be the error class
    print(y_trial.shape)
    #type(y_trial)
   
   #adding 3 blank rows for the 3 error classes that never come 
    row,col=y_trial.shape
    blank_col= np.zeros(row)
    y_trial= np.insert(y_trial,32,blank_col, axis=1)
    y_trial= np.insert(y_trial,35,blank_col, axis=1)
    y_trial= np.insert(y_trial,37,blank_col, axis=1)
    
    y_pred= np.zeros((n,k))
    
    #keeping the top 5 predicted classes only
    for i in range(n):
        y_pred[i,:]=np.argsort(y_trial[i])[49:44:-1]
        y_pred[i,:]= y_pred[i,:] + 1
        
    return y_pred
