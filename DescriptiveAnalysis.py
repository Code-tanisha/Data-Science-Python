import os
import pandas as pd

os.chdir("C:\Python")
data = pd.read_csv("iris.csv")
data.head()

data.info()
data['Sepal.Length'].sum()

# Mean Petal.width
data['Petal.Width'].mean()

# Cumulative sum of petal.width
data['Petal.Width'].cumsum()

# Summary Statistics 
data['Petal.Width'].describe()

# Count the no of non-NA values
data['Petal.Width'].count()

# Minimum value of petal.width
data['Petal.Width'].min()
data['Petal.Width'].max()

# median
data['Petal.Width'].median()

# Sample variance
data['Petal.Width'].var()

# std deviation
data['Petal.Width'].std()

# skewness of petal.width
data['Petal.Width'].skew()

# kurtosis
data['Petal.Width'].kurt()

# correlatiob matrix of values
data.corr()

# covariance matrix of values
data.cov()
