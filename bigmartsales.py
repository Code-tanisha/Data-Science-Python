# -*- coding: utf-8 -*-
"""BigMartSales.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18F2kL5kkcwQ9dfFoHLALStceA0ZEwmEy

PROBLEM STATEMENT :-
BigMart sales dataset consists of 2013 sales data for 1559 products across 10 different outlets in different cities. The goal of the BigMart sales prediction ML project is to build a regression model to predict the sales of each of 1559 products for the following year in each of the 10 different BigMart outlets. The BigMart sales dataset also consists of certain attributes for each product and store. This model helps BigMart understand the properties of products and stores that play an important role in increasing their overall sales.
"""

# Commented out IPython magic to ensure Python compatibility.
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from google.colab import files
uploaded = files.upload()

df = pd.read_csv(io.BytesIO(uploaded['Train_UWu5bXk.csv']))

from google.colab import files
uploaded = files.upload()

df = pd.read_csv(io.BytesIO(uploaded['Test_u94Q5KV.csv']))

"""UNDERSTANDING THE DATA"""

train  = pd.read_csv("Train_UWu5bXk.csv")
test   = pd.read_csv("Test_u94Q5KV.csv")

train.head()

train.shape

test.head()

test.shape

print("the shape of the train data is :", train.shape)
print("the shape of the test data is :", test.shape)

train.dtypes # Checking the datatypes in training dataset

test.dtypes # Checking the datatypes in test dataset

train.describe()

test.describe()

"""**PreProcessing**"""

train.head()

train.Item_Fat_Content.value_counts()

train.Item_Type.value_counts()

train.Outlet_Size.value_counts()

train.Outlet_Location_Type.value_counts()

train.Outlet_Type.value_counts()

"""**DATA CLEANING**"""

#Combine test and train into one file
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True, sort = False)
print(train.shape, test.shape, data.shape)

# Check missing values
data.apply(lambda x: sum(x.isnull()))

"""Filling missing values"""

data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())

data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())

data['Outlet_Size'].value_counts()

data.Outlet_Size = data.Outlet_Size.fillna('Median')

data.apply(lambda x: sum(x.isnull()))

data.info()

data.head()

"""**Numeric and One HOT coding of categorical value**"""

#Item type combine:
data['Item_Identifier'].value_counts()
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

# Import library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
# New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Outlet']
le = LabelEncoder()
for i in var_mod:
  data[i] = le.fit_transform(data[i])

# One Hot Coding:
data = pd.get_dummies(data, columns = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Type_Combined', 'Outlet'])

data.head()

data.dtypes

"""**Exporting the data**"""

# Drop the columns which have been converted to different types
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
# Divide into test and train
train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']

# Drop unessary columns
test.drop(['Item_Outlet_Sales', 'source'], axis = 1, inplace = True)
train.drop(['source'], axis = 1, inplace = True)
# Export files as modified versions
train.to_csv("train_modified.csv", index = False)
test.to_csv("test_modified.csv", index = False)

"""**Model Building**"""

# Reading modified data
train2 = pd.read_csv("train_modified.csv")
test2 = pd.read_csv("test_modified.csv")

train2.head()

X_train = train2.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
y_train = train2.Item_Outlet_Sales

test2.head()

X_test = test2.drop(['Outlet_Identifier', 'Item_Identifier'], axis = 1)

X_train.head()

y_train.head()

"""**Linear Regression Model**"""

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(X_test)
y_pred

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

lr_accuracy = round(regressor.score(X_train, y_train) * 100,2)
lr_accuracy

r2_score(y_train, regressor.predict(X_train))

submission = pd.DataFrame({
'Item_Identifier':test2['Item_Identifier'],
'Outlet_Identifier':test2['Outlet_Identifier'],
'Item_Outlet_Sales': y_pred
},columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])

submission.to_csv('submission1.csv',index=False)