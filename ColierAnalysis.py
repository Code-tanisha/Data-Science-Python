import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir("C:\Python")
os.getcwd()
marketing_train = pd.read_csv("marketing_tr.csv")
marketing_train
marketing_train.head()

#Here we will copy our data to some object for analysis
df = marketing_train.copy()

###OUTLIER ANALYSIS

# Plot boxplot to visualize
#plt.boxplot is a function helps in ploting
#me need to give a variable name for which we need to plot the graph so custAge is variable here.

%matplotlib inline
plt.boxplot(marketing_train['custAge'])

#Save numeric names 
#here we are saving all the variable names which are continuous becoz as we know that outlier analysis is only applied on continuous

cnames = ["custAge", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "euriborin", "nr.employed", "pmonths", "pastEmail"]
cnames

#Detect and delete outlies from data.
#Here we need to calculate the 25% of a particular variable here we also need to calculate the 75 percentile of a continuous variable
#to develob a boxplot outer fence and inner fence and then we need to map all the values to the boxplot to know whether any values is falling 
#beyond upper fence and lower fence which will be considered as an outlier
#np.percentile is the function which hels us to calculate the percentile of the variable
#iqr = iqr is interquartile rance which helps us to calculate the lower and upper fence
for i in cnames:  
    print(i)
    q75, q25 = np.nanpercentile(marketing_train.loc[:,i], [75,25])
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5) #Lower fence
    max = q75 + (iqr*1.5) #Upper fence
    print(min)
    print(max)
#here we need to write line of code which says drop those observation which are less than lower fence and which are higher than upper fence
    marketing_train = marketing_train.drop(marketing_train[marketing_train.loc[:, i] < min].index)
    marketing_train = marketing_train.drop(marketing_train[marketing_train.loc[:, i] > max].index)
 marketing_train.shape
 
 #EXPERIMENT2
#Here we will try to replace outliers with NAs and then will detect missing values
# nanpercentile is used to compute the nth percentile
q75, q25 = np.nanpercentile(marketing_train['custAge'], [75,25])
q75

iqr  = q75 - q25

#calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace all the values beyond upper fence and lower fence with nan
#custAge is variable which are we concerned here from
marketing_train.loc[marketing_train['custAge'] < minimum,:'custAge'] = np.nan
marketing_train.loc[marketing_train['custAge'] > maximum,:'custAge'] = np.nan


#Calculate missing value
#function to calculate missing value
missing_val = pd.DataFrame(marketing_train.isnull().sum())
missing_val

#Impute KNN imutation
#To impute missing values present in the customer Age
marketing_train = pd.DataFrame(KNN(k = 3).complete(marketing_train),columns = marketing_train.columns)



