import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaboarn as sns
from random import randrange, uniform
os.chdir("C:\Python")
marketing_train = pd.read_csv("marketing_tr.csv")
marketing_train.head()

## MISSING VALUE ANALYSIS
#we need to find how many missing values are there in each variable
#is cell is the function which identify whether this cell heve missing values or not
missing_val = pd.DataFrame(marketing_train.isnull().sum())
missing_val

# Reset index
# Let us create the table in a prober way so that we can use the index in proper way
missing_val = missing_val.reset_index()
missing_val

# Rename is a function and what it will do is it will change the the name of the variable
# here we want to change the name of the variable index
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
missing_val

# Calculate percentage
# here we will convert the absolute values to percentage
missing_val['Missing Percentage'] = (missing_val['Missing_percentage']/len(marketing_train))*100

# Here we will sort the data in descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
# Save output results in csv format
missing_val.to_csv("Missing_perc.csv", index = False)

# Create missing value
# we have selected a variable where we will wantedly create the missing value custAge
# np.nan will convert the value which is present at 70th position to missing value
marketing_train['custAge'].loc[70] = np.nan

marketing_train['custAge'].loc[70]
# Impute with mean
# fillna = this function will see where the missing values are there in this variable and the compute this with mean

marketing_train['custAge'] = marketing_train['custAge'].fillna(marketing_train['custAge'].mean())

# Impute with median
# Again after creating missing values then only impute the median
marketing_train['custAge'] = marketing_train['custAge'].fillna(marketing_train['custAge'].median())
marketing_train['custAge'].loc[70]

# again create the missing value and then
# in this iteration we will impute KNN Imputation
# KNN Imputation method - what it will do is KNN will calculate the distance between observation
# So to calculate the distance we need to convert the character value into the numerical values let suppose we have a variable gender it hase to categories so instead of male we will write 0 and instead of female we can write female
# fit_transform is a function of KNN library whuch takes the data as input argument
marketing_train = pd.DataFrame(KNN(k=3).fit_transform(marketing_train), columns = marketing_train.columns)

