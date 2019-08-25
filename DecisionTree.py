import os
import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

os.chdir("C:\Python")
marketing_train = pd.read_csv("marketing_tr.csv")
marketing_train.head()
df = marketing_train.copy()
df.dtypes
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

# We need to clean null values
obj_df[obj_df.isnull().any(axis=1)]

obj_df["schooling"].value_counts()
# we will replace null values. so after this data will not be having any null values
obj_df = obj_df.fillna({"schooling" : "four"})
obj_df = obj_df.fillna({"day_of_week" : "four"})

# So fromhere data does not have any null values so we can look at options for encoding
obj_df["marital"].value_counts()

cleanup_nums = {"marital": {"married": 1, "single": 2, "divorced": 0, "unknown": 8},
               "schooling": {"four": 4, "university.degree": 1, "high.school": 2, "basic.9y": 3, "professional.course": 5, "basic.4y": 6, "basic.6y": 7, "unknown": 8, "illiterate": 9},
               "profession": {"admin.": 0, "blue-collar": 1, "technician": 2, "services": 3, "management": 4, "retired": 5, "entrepreneur": 6, "self-employed": 7, "housemaid": 8, "unemployed": 9, "student" : 10, "unknown": 11},
               "housing": {"no": 0, "yes": 1, "unknown": 2}, "loan": {"no": 0, "yes": 1, "unknown": 2}, "responded": {"no": 0, "yes": 1},
               "contact": {"cellular": 0, "telephone": 1}, "month": {"may": 0, "jul": 1, "aug": 2, "jan": 3, "nov": 4, "apr": 5, "oct": 6, "sep": 7, "mar": 8, "dec": 9, "jun": 10},
               "day_of_week": {"mon": 0, "thu": 1, "tue": 2, "wed": 3, "fri": 4, "four": 5}, "poutcome": {"nonexistent": 0, "failure": 1, "success": 2},
               "default": {"no": 0, "unknown": 2, "yes": 1}}
obj_df.replace(cleanup_nums, inplace=True)
obj_df.head()

obj_df1 = df.select_dtypes(include=['int64']).copy()
obj_df1.head()

df3 = pd.concat([obj_df, obj_df1], axis = 1)
df3

#Divide the data into train and test
# all the independent values into x axis and all the dependent values into y axis
# here responded is our dependent variable
# here we ave 15 independent variable thats why provided index 0 to 16
x = df3.values[:, 0:15]
# here our dependent variable is responded so 16 is its index
y = df3.values[:, 14]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

type(y_train)

clf = tree.DecisionTreeClassifier(criterion= 'entropy').fit(x_train, y_train)

clf.fit(x_train, y_train)

#predict new test cases
y_pred = clf.predict(x_test)

y_pred

# create dot file to visualize tree #http://webgraphviz.com
# 1st we are saving file in the form of pt.dot file, and then we need to paste all this file to website to generate the plot

dotfile = open("pt.dot", 'w')
df = tree.export_graphviz(clf, out_file=dotfile, feature_names = df3.columns)
# copy the full code from here and paste that to above website to form a decision tree

#check accuracy of model
accuracy_score(y_test, y_pred)*100

REGRESSION

df = pd.read_csv("birthwt.csv")
df.shape

# Divide data into train and test
# train_test_split =helps to divide the data in 2 subsets
# test_size=0.2 means that the 20 % of the data will go into the test and the remaining will go in training model
train, test = train_test_split(df, test_size=0.2)
train.shape

# Decision tree for regression on the training data
# max_deplt = 2 means max node for a particular leaf will be 2
# under the fit we need to provide the name of the model, so here we are building a model on the training data so 1st argument is independent variable ie.0;9 -it will select 0 to 8 variable
# 2nd argument = dependent variable last variable  the 9 th variable is the dependent variable
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9], train.iloc[:, 9])

fit_DT

# Apply decision tree on the test data
# Predict is a function wich is used to predict on the unseen data which is our test data
# here in test.iloc we need to select only the independent variable.
# we are not including the dependent variable because we are feeding test data to the model to predict to predict their target value
prediction_DT = fit_DT.predict(test.iloc[:, 0:9])

prediction_DT
# there are all our test case values or predicted values 
# so from this we can compare the actual values vs presicted values to look at the percentage of error and the accuracy we get 
