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

# Step 2 - Build Deep Learning Model
"""
import keras library , by default keras uses 
TensorFlow as backend, ( alternative: Theano)
Sequential package :  Initialize ANN
Dense package : Build layers of ANN
"""
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize ANN
nn_classifier =  Sequential()

# Add input layers and hidden layers with rectifier AF
nn_classifier.add(Dense(units = 6, #formerly output_dim
                        input_dim = 11,
                        kernel_initializer = "uniform",
                        activation = "relu"))
# Second layer
nn_classifier.add(Dense(units = 6, #formerly output_dim
                        kernel_initializer = "uniform",
                        activation = "relu"))

# Output layer : sigmoid activation (binary Output)
nn_classifier.add(Dense(units = 1,
                        kernel_initializer = "uniform",
                        activation = "sigmoid"))

# Compile NN : Apply stochastic GD
nn_classifier.compile(optimizer = "adam",
                      loss = "binary_crossentropy",
                      metrics = ["accuracy"])

# Fitting the ANN
nn_classifier.fit(X_train, y_train, batch_size = 10,
                  epochs = 100)


y_pred = nn_classifier.predict(X_test)

# y_pred is in probablities, convert it to binary 0, 1
y_pred = ( y_pred > 0.5 )

# Given a customer! Scaled 
new_prediction = nn_classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 
                                              60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# make Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)