"""
churn problem : WHICH CUSTOMER WILL LEAVE THE BANK :( ?
: Given the dataset of 10k bank customers, 
: build ANN to find customer with the 
: following features is likely to leave 
: the bank and evaluate the model performance.
: Geography : "France"
: Credit Score : 600
: Gender : male
: Age : 40yr old
: Tenure: 3yrs
: Balance :  $60k
: Number of products : 2
: Holds creditcard
: An active user
: estimated Salary : $50k
"""
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
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
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
    classifier.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10,
                             epochs = 100)
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()



