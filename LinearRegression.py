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
df = marketing_train.copy()
df.dtypes
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

# We need to clean null values
obj_df[obj_df.isnull().any(axis=1)]

# we will replace null values. so after this data will not be having any null values
obj_df = obj_df.fillna({"schooling" : "four"})
obj_df = obj_df.fillna({"day_of_week" : "four"})

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

ERROR METRICS

# Built confusion_matrix
#confusion_matrix helps us to built a contegency table or confusion matrix
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
CM

CM = pd.crosstab(y_test, y_pred)
CM

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

REGRESSION

df = pd.read_csv("birthwt.csv")

# DIVIDE DATA INTO TRAIN AND TEST
train, test = train_test_split(df, test_size = 0.2)

fit_DT = fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9], train.iloc[:, 9])

fit_DT

prediction_DT = fit_DT.predict(test.iloc[:, 0:9])

# As we know that at very high level there are 3-4 types og regression metics MAE, MAPE, RME, RMSE
# Calculate MAPE(mean absolute percentage error = actual-predicted/actual)
# abs = it will take round of all the values
def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# CALLING OF ABOVE FUNCTION
# test.iloc[:, 9] = actual values, prediction_DT = predicted values
MAPE(test.iloc[:, 9], prediction_DT)

# Linear regression can be applied only on the continuous dependent variable

import statsmodels.api as sn
# Train the model using the training sets
# ols stands for optimum least square,it helps us to built the linear regression model using the least sq meathod
# train.iloc[:,9] = dependent variables on 9th location, train.iloc[:, 0:9] = independent variables
model = sn.OLS(train.iloc[:,9], train.iloc[:, 0:9]).fit()

# Print summary satatistics
# it helps us to see the different parameters which helps in building training data
model.summary()

# make the predictions by the model
predictions_LR = model.predict(test.iloc[:,0:9])
predictions_LR

# calculate MAPE
MAPE(test.iloc[:,9], predictions_LR)



