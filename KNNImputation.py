import os
import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

os.chdir("C:\Python")
marketing_train = pd.read_csv("marketing_tr.csv")
marketing_train.head()
df = marketing_train.copy()
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

# We need to clean null values
obj_df[obj_df.isnull().any(axis=1)]

obj_df["schooling"].value_counts()

# we will replace null values. so after this data will not be having any null values
obj_df = obj_df.fillna({"schooling" : "four"})
obj_df = obj_df.fillna({"day_of_week" : "four"})

# So from here data does not have any null values so we can look at options for encoding
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

clf = tree.DecisionTreeClassifier(criterion= 'entropy').fit(x_train, y_train)

clf.fit(x_train, y_train)

#predict new test cases
y_pred = clf.predict(x_test)

# create dot file to visualize tree #http://webgraphviz.com
# 1st we are saving file in the form of pt.dot file, and then we need to paste all this file to website to generate the plot

dotfile = open("pt.dot", 'w')
df = tree.export_graphviz(clf, out_file=dotfile, feature_names = df3.columns)
# copy the full code from here and paste that to above website to form a decision tree

#check accuracy of model
accuracy_score(y_test, y_pred)*100

# Built confusion_matrix
#confusion_matrix helps us to built a contegency table or confusion matrix
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)

CM = pd.crosstab(y_test, y_pred)

# Let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

# cheq accuracy of model
accuracy_score(y_test, y_pred)*100

# Accuracy
((TP+TN)*100)/(TP+TN+FP+FN)

# False negative rate
# this is our imp erroe rate
# WE need to reduce the false negative as much as we can
(FN*100)/(FN+TP)

# Recall
(TP*100)/(TP+FN)

# Random forest
# 1st argument = n_estimators=100(no of trees to be developed should be predefined) 

RF_model = RandomForestClassifier(n_estimators=500).fit(x_train, y_train)

RF_model

# we are applying the same random forest model which we have built on the train data
# RF_model is the odel which we have built on the train data , predict is a function which helps us to take the input on the new test data without target value and then predict its new target value using the provided model
RF_Predictions = RF_model.predict(x_test)

RF_Predictions

# to evaluate the performance of any classification model we need to go for confusion matrix
# built a confusion matrix
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
# here we are building a contigency tables from the actual values vs the predicted values as arguments are passed

CM = pd.crosstab(y_test, RF_Predictions)

CM = pd.crosstab(y_test, RF_Predictions)

# Let us save TP, TN, FN, FP
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

# cheq accuracy of the model
# accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

# False negative rate
#(FN*100)/(FN+TP)

KNN IMPUTATION

# KNN implementation
# here we are starting with k = 1 then we will cheq further later
# n_neighbors is the value of k and we can take only odd values here and cheq accuracy. We can not take even values here
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(x_train, y_train)

# Predict the test case
KNN_Predictions = KNN_model.predict(x_test)

KNN_Predictions

# Built the confurion matrix
CM = pd.crosstab(y_test, KNN_Predictions)

# Let us save TP, TN, FN, FP
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

# cheq accuracy of the model
# accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

# False negative rate
#(FN*100)/(FN+TP)
