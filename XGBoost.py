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

imputer = Imputer(missing_values = -1, strategy = 'most_frequent', axis = 0)
X[:,0:20] = imputer.fit_transform(X[:, 0:20])

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
X[:,20:22] = imputer.fit_transform(X[:, 20:22])

imputer = Imputer(missing_values = -1, strategy = 'most_frequent', axis = 0)
X[:,22:33] = imputer.fit_transform(X[:, 22:33])

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
X[:,33:37] = imputer.fit_transform(X[:, 33:37])

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
X[:,37:58] = imputer.fit_transform(X[:, 37:58])

####################################################################

imputer = Imputer(missing_values = -1, strategy = 'most_frequent', axis = 0)
Z[:,0:20] = imputer.fit_transform(Z[:, 0:20])

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
Z[:,20:22] = imputer.fit_transform(Z[:, 20:22])

imputer = Imputer(missing_values = -1, strategy = 'most_frequent', axis = 0)
Z[:,22:33] = imputer.fit_transform(Z[:, 22:33])

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
Z[:,33:37] = imputer.fit_transform(Z[:, 33:37])

imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
Z[:,37:58] = imputer.fit_transform(Z[:, 37:58])


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


# Data standardisation

sc = StandardScaler()
X  = sc.fit_transform(X)
Z  = sc.fit_transform(Z)

# PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.99)
X_decom = pca.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.99)
Z_decom = pca.fit_transform(Z)

# Splitting into train and test set

X_train, X_test, y_train, y_test = train_test_split(X_decom,y,test_size = 0.20)

# define gini function

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

# prediction
import xgboost as xgb
params = {
    'min_child_weight': 10,
    'objective': 'binary:logistic',
    'max_depth': 5,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700
    }
# Convert our data into XGBoost format
d_train = xgb.DMatrix(X_train,y_train)
d_valid = xgb.DMatrix(X_test, y_test)
d_test = xgb.DMatrix(Z_decom)
watchlist = [(d_train, 'train'), (d_valid, 'test')]
    
clf = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, 
                feval= gini_xgb, maximize=True, verbose_eval=100)
y_pred = clf.predict(d_test)


