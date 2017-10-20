# importing libraries

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


# importing dataset

dataset = pd.read_csv('train.csv')
dataset_val = pd.read_csv('test.csv')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, [1]].values
y = y.ravel()
Z = dataset_val.iloc[:,1:].values

# imputing missing values

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
X[:,0:57] = imputer.fit_transform(X[:, 0:57])

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
Z[:,0:57] = imputer.fit_transform(Z[:, 0:57])

# One hot Encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [7])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [8])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [9])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [11])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [12])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [13])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [14])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [15])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [16])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [17])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [18])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [19])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [21])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [22])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [23])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [24])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [25])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [26])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [27])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [28])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [29])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [30])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [31])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [32])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [37])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [38])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [40])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [41])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [42])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [43])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [44])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [45])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [46])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [47])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [48])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [49])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [50])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [51])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [52])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [53])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [54])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [55])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [56])
X = onehotencoder.fit_transform(X).toarray()

# Z encoding

onehotencoder = OneHotEncoder(categorical_features = [0])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [1])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [2])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [3])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [4])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [5])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [6])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [7])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [8])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [9])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [10])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [11])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [12])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [13])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [14])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [15])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [16])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [17])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [18])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [19])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [21])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [22])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [23])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [24])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [25])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [26])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [27])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [28])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [29])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [30])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [31])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [32])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [37])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [38])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [40])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [41])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [42])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [43])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [44])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [45])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [46])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [47])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [48])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [49])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [50])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [51])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [52])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [53])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [54])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [55])
Z = onehotencoder.fit_transform(Z).toarray()

onehotencoder = OneHotEncoder(categorical_features = [56])
Z = onehotencoder.fit_transform(Z).toarray()

# Feature Scaling

sc = StandardScaler()
X = sc.fit_transform(X)
Z = sc.fit_transform(Z)  
# Separate Test/Train Set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1)
    
# Stochastic Gradient Boosting Classification
from sklearn.ensemble import GradientBoostingClassifier
g_clf = GradientBoostingClassifier(n_estimators=500, verbose = 2)
g_clf.fit(X_train, y_train)
    
############ Run for test set-------------------> RunTHIS!!!
g_clf = GradientBoostingClassifier(n_estimators=500, verbose = 2)
g_clf.fit(X, y)
    
# predict X_test
    
y_pred = g_clf.predict_proba(X_test)
y_pred = y_pred[:,1]
y_test = y_test.ravel()
    
 # Gini coeffient
def Gini(y_test, y_pred):
    # check and get number of samples
    assert y_test.shape == y_pred.shape
    n_samples = y_test.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_test, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

Gini = Gini(y_test, y_pred)


######################################

# Importing the data 

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, 1:].values

# Imputing missing values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
X_test[:,0:57] = imputer.fit_transform(X_test[:, 0:57])

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

# predict X_test

y_pred = g_clf.predict_proba(X_test)
y_pred = y_pred[:,1]
