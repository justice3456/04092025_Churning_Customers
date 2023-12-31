# -*- coding: utf-8 -*-
"""Churning Customers in a Telecoms Company.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tt_Y6PpKsLAjHo2qLtlNy8DUFcG51UGT
"""

#starting the project
import os
import sklearn
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import pickle

""" **Loading The Data and Analysing the Datat**"""

#setting up google drive
from google.colab import drive
drive.mount('/content/drive')

#importing the dataset
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Intro_to_AI/CustomerChurn_dataset.csv')

data.head()

#trying to convert total charges to string failed because there are some missing values in total charges.
#the first instance of a missing value is at position '488'
data.iloc[488]

#imputing and converting total charges from object to numeric
data['TotalCharges'] = data['TotalCharges'].fillna(method='ffill')
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce', downcast = 'float')

data.iloc[488]

#there are no other missing values thus there is no need to impute
#customerID is removed because the ID has no importance in predicting customer churn
data.drop('customerID', axis = 1, inplace = True)

#numeric data does not need to be encoded thus, 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges' are not encoded
data_to_encode = data.drop(['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'], axis = 1)

from sklearn.preprocessing import LabelEncoder

#data_to_encode is a part of the dataset that needs to be encoded and the have non numeric datatypes
data_to_encode.info()

#creating an function (automation) for encoding a dataframe which returns an encoded dataframe
label_encoder = LabelEncoder()
def AutomaticLabelEncoding(dataframe):
  encoded_columns = pd.DataFrame() #this dataframe is updated with encoded values during every iteration
  for i in dataframe:
    i = str(i)
    label_encoder_ = LabelEncoder()
    encoding = pd.Series(label_encoder.fit_transform(dataframe[i]), name = i)
    encoded_columns[i] = encoding
  return encoded_columns

#encodind the data
encoded_dataframe = AutomaticLabelEncoding(data_to_encode)

encoded_dataframe.info()

path_encoder = '/content/drive/My Drive/Colab Notebooks/Intro_to_AI/Churning_encoder.pkl'
with open(path_encoder, 'wb') as f:
    pickle.dump(label_encoder, f)

#a function that takes two dataframes and concats the second dataframe(datafram_to_add) to the right hand side of the first dataframe(dataframe)
def SimpleConcat(dataframe, dataframe_to_add):
  for i in dataframe_to_add:
    dataframe[str(i)] = dataframe_to_add[str(i)]
  return dataframe

#concatinating the encoded data with the data that did not need encoding
numeric_data = SimpleConcat(encoded_dataframe,data[['SeniorCitizen', 'tenure', 'MonthlyCharges']] )

#A complete dataframe where all values are numeric as it is a requirement for Machine Learning
numeric_data.info()

#defining dependent and independent columns
y = numeric_data['Churn']
X = numeric_data.drop('Churn', axis = 1)

from sklearn.preprocessing import StandardScaler

# #Scaling independent columns
# Xscaled = StandardScaler().fit_transform(X.copy())
# X = pd.DataFrame(Xscaled, columns = X.columns)

scaler = StandardScaler().fit(X.copy())
X_scaled = scaler.transform(X.copy())
X = pd.DataFrame(X_scaled, columns=X.columns)

X.info()

#Saving scaler
path_scaler = '/content/drive/My Drive/Colab Notebooks/Intro_to_AI/Churning_scaler1.pkl'
with open(path_scaler, 'wb') as f:
    pickle.dump(scaler, f)

data = pd.concat([X, y], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

"""**(1) Using a well known method to extract relevant features that relate to chuning**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Create a tree-based model (Random Forest in this example)
model = RandomForestClassifier(n_estimators = 150, random_state = 42)

feature_vetting = RFECV(estimator = model, step = 1, cv = 3, scoring = 'accuracy')
feature_vetting.fit(X_train, y_train)

randomforest_features = X_train.columns[feature_vetting.support_]

#Using correlation to extract relevant features
correlations = numeric_data.corr(numeric_only = True)
correlations['Churn'].sort_values(ascending = False)

abs_correlations = correlations['Churn'].abs()
selected_features = correlations['Churn'].abs()[correlations['Churn'].abs() > 0.19].index.tolist()
selected_features = [col for col in selected_features if col in randomforest_features]

#selected features
selected_features

"""**(2) Using EDA to find out which customer profiles relate to EDA a lot**"""

import seaborn as sns

sns.boxplot(x = 'Churn', y ='tenure', data = numeric_data.copy())
plt.show()
print('\n')

for i in range(0,7):
  sns.countplot(x = numeric_data.copy().columns[i], hue='Churn', data = numeric_data.copy())
  plt.show()
  print('\n')

#splitting the dataset again to since the dataset has been updated to relevant features and the outcome
X = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

"""**(3) Using the features in (1) to define and train a Multi-Layer Perceptron model using the Functional API**"""

#Functional API
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer_1 = Dense(200, activation='relu')(input_layer)
hidden_layer_2 = Dense(130, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(87, activation='relu')(hidden_layer_2)
hidden_layer_4 = Dense(25, activation='relu')(hidden_layer_3)
hidden_layer_5 = Dense(10, activation='relu')(hidden_layer_4)
# Define the output layer for binary classification
output_layer = Dense(1, activation='sigmoid')(hidden_layer_5)

#Model
model = Model(inputs = input_layer, outputs = output_layer)

#Optimizing the model
model.compile(optimizer = Adam(learning_rate = 0.00001), loss = 'binary_crossentropy', metrics = ['accuracy'])
model_returned = model.fit(X_train, y_train, epochs = 12, batch_size = 10, validation_data = (X_test, y_test))

"""**Evaluate the model’s accuracy and calculate the AUC score**"""

# Plotting training and validation loss across epochs
train_loss = model_returned.history['loss']
val_loss = model_returned.history['val_loss']
train_acc = model_returned.history['accuracy']
val_acc = model_returned.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label ='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy across epochs
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

"""**Evaluating the model’s accuracy, calculating the AUC score**"""

val, accuracy = model.evaluate(X_train, y_train)
accuracy*100

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.4f}')

from sklearn.metrics import roc_auc_score

y_pred_proba = model.predict(X_test)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f'AUC Score: {auc_score:.4f}')

"""**Training an MLP using features from (1) with cross validation and GridSearchCV**"""

mlp_crossvalidation = MLPClassifier(max_iter = 100, hidden_layer_sizes = (50,))

# Perform cross-validation
scores = cross_val_score(mlp_crossvalidation, X, y, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())

# Define the MLPClassifier
mlp_gridsearch = MLPClassifier(max_iter = 100)

# Define the parameter grid to explore different hyperparameters
parameter_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'batch_size': [10,32,64],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Using GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(mlp_gridsearch, parameter_grid, cv = 5, scoring = 'accuracy', verbose = 1)

# Fit the model on the training data
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_model.score(X_test, y_test)

print("Best Hyperparameters:", best_params)
print("Test Accuracy:", test_accuracy)

"""**Optimiseing the model, training and testing again**"""

model_returned = model.fit(X_train, y_train, epochs = 12, batch_size = 64, validation_data = (X_test, y_test))

# Plotting training and validation loss across epochs
train_loss = model_returned.history['loss']
val_loss = model_returned.history['val_loss']
train_acc = model_returned.history['accuracy']
val_acc = model_returned.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label ='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy across epochs
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

val, accuracy = model.evaluate(X_train, y_train)
accuracy*100

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.4f}')

y_pred_proba = model.predict(X_test)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f'AUC Score: {auc_score:.4f}')

"""**Saving the model**"""

path_model = '/content/drive/My Drive/Colab Notebooks/Intro_to_AI/Churning_model1.pkl'
with open(path_model, 'wb') as f:
    pickle.dump(model_returned, f)