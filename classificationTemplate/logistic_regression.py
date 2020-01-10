# Data preprocessing 

# Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# I - import Datasets 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# IV - Crossvalidation sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 0)

# V - feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# model logistic regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0) 

# train
lr_classifier.fit(X_train, y_train)

# predict
y_pred = lr_classifier.predict(X_test)

# Evaluation : confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

