# Achieves about 85.17% accuracy.
# TODO : modify to reach 86% accuracy

import pandas as pd
import numpy as np
import matplotlib as plt

# step 1 - import dataset and pre-process the data
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# if there were catagorical features in X
from sklearn.preprocessing import LabelEncoder
# Country field at index 1
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
# gender field at index 2
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columntransformer = ColumnTransformer([("Geography", OneHotEncoder(), [1])],
                                       remainder = 'passthrough')

X = np.array(columntransformer.fit_transform(X), dtype='float64')
X = X[:, 1:]

# split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0)

# feature scaling - Standardize 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,
                         kernel_initializer = 'uniform',
                         activation = 'relu',
                         input_dim = 11))
    classifier.add(Dense(units = 6,
                         kernel_initializer = 'uniform',
                         activation = 'relu')) 
    classifier.add(Dense(units = 1,
                         kernel_initializer = 'uniform',
                         activation = 'sigmoid')) 
    classifier.compile(optimizer = optimizer,
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
# Tuning hyper parameters : epoch, optimizer, batch size
parameters = {'batch_size': [25, 50],
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_






