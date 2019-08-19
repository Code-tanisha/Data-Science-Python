##
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

from scipy.stats import chi2_contingency
os.chdir("C:\Python")
marketing_train = pd.read_csv("marketing_tr.csv")

#these cnames are our continuous variables
cnames = ["custAge", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "euriborin", "nr.employed", "pmonths", "pastEmail"]
cnames

#correlation analysis is applied only on numerical data
#so lets create a subset of data which contains only continuous variabl
df_corr = marketing_train.loc[:,cnames]

#using names of cnames we will subset the data 
#correlation analysis is applied only on numerical data
#so lets create a subset of data which contains only continuous variabl
df_corr = marketing_train.loc[:,cnames]

#we have here 10 variables as continuous variable
df_corr.shape

#Develop a correlation plot on top of it
#Set the width and height of the plot
#plt has got a function call subplot which will set the height and width
#height = 7, width = 5
f, ax = plt.subplots(figsize = (7,5))

# Generate a correlation matrix

#df_corr is the data set where all the continuous variables are stored

# corr() is a built in function which develops a correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
# sns= seaborn library; heatmap is a function present in sns library and it helps us to plot a correlation plot
# mask will create the individual blocks of the correlation matrix
# np.zeros_like will create a fix not of slots or square brackets under the whole plane
# cmaps is used to set the colors in correlation matrix
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square = True, ax=ax)

#red color= highly positively correlated data
#blue color = highly negatively correlated data
#ex-custage has no dependency
#according to graph we can see that emp.var.rate and pdays are highly correlated with other variables . so we can drop these 2 variable

## chisquare Test
# we will see that how to get a redundant variable from categorical variable
# here we will save all the categorical values
cat_names = ["profession", "marital", "schooling", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome", "responded"]
cat_names

# Loop for chi square test
# in each iteration each variable will pass and it will develop the chi sq test and generate the output and send the p values
# chi2_contingency is imported from scipy library an this will helps to dvelos chi sq test
# input to chi sq test is alwas a contingency table , so here we are developing a contingency table using crosstab
# cross tab is a function available in pandas library
#argument to cross tab is 1st is the target variable or dependent variable, which is marketing_train, and the 2nd is independent variable marketing train[i], i as we are dealing in the loop so marketing train of i
# This will give us 4 values 1: chi2 values 2: p-value 3: degrees of freedon value dof 4: expected value ex
for i in cat_names:
    print(i)
    chi2, p , dof, ex = chi2_contingency(pd.crosstab(marketing_train['responded'], marketing_train[i]))
    print(p)
    
# Depending on the p values which we got above we will reject the hypothesis
# if the p vlue is less than 0.05 then reject the null hypothesis
# in above only housing and loan has value greater than 0.05 so drop them
# axis = 1 coz we are dealing with column level data
marketing_train = marketing_train.drop(['pdays', 'emp.var.rate', 'day_of_week', 'loan', 'housing'], axis=1)

marketing_train.shape

########## FEATURE SCALING
marketing_train = pd.read_csv("marketing_tr.csv")

marketing_train.head()

df = marketing_train.copy()

# plot the histogram to look whether the data is normally distributed or not
# this is our 1st cheq to understand the data behaviour
#hist is the function present in matplot lib to plot histogram
#1st  variable is marketing trand of custAge, we know that histogram works on the bins
#it will plot the bins based on the intervals. auto bins means we ane not providing any no of bins.depending on the data it will take automatically
# we can take instead of compaign , customerage and cheq whether the data is normally distributed or not
#here data is not normally distributer it is left skewed so we will aplly normalizatio
%matplotlib inline
plt.hist(marketing_train['campaign'], bins = 'auto')

cnames = ["custAge", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "nr.employed", "pmonths", "pastEmail"]


# Normalizatiom
# we will use the loop to iterate over each continuous variable to convert the data into same range
for i in cnames:
    print(i)
    marketing_train[i] = (marketing_train[i] - min(marketing_train[i]))/(max(marketing_train[i]) - min(marketing_train[i]))
    
 marketing_train.head(10)
 
 
#look at the range of each variable coz according to the formula the range should be between 0 to 1
marketing_train['custAge'].describe()

marketing_train['campaign'].describe()

#here after normalization we will take the original data to apply standardization
df = marketing_train.copy()


#Standardization
#
for i in cnames:
    print(i)
    marketing_train[i] = (marketing_train[i] - marketing_train[i].mean())/marketing_train[i].std()

marketing_train.head()

## SAMPLING TECHNIQUES

marketing_train.shape

#Simple random sampling
#it wil just randomly pick the predefined no of observation from the raw data
#sample is a function which just pich the predefined no of functions from the master data
#extract no of observtion we want ie. 100 from the sample data
Sim_Sampling = marketing_train.sample(100)
Sim_Sampling.shape
Sim_Sampling.head(10)

## Systematic Sampling - 1st it will calculate the K-value
# Depending on the k value it will select every kth observation from the data set
#calculate k value; K = N/n  ; N = len(marketing_train) total no of observation; n = the amont of observation u want in a sample data

k = len(marketing_train)/3500
print(round(k))

#Generate a random number using simple random sampling
# after selecting k value the next step is how K value will start selecting observations
#in this the 1st observation it randomly pick eg. row number 5 then the next will be selected according to the kth value. suppose k = 2, so it will pick 7th row then 9th
RandNum = random.randint(0, 5)
RandNum

# Select kth observation starting from RandNum
# iloc is to specify the location of the observation
# only colon : after comma means select all the variables
Sys_Sampling = marketing_train.iloc[RandNum::k, :] # every kth row , but starting at RandNum
#After starting from the RandNum the kth value ie 2 will be selected

Sys_Sampling = marketing_train.iloc[RandNum::k, :]
Sys_Sampling.head()

### Sratified sampling
# what we do in this testing is we will provide one categorical variable and based on this categorical variable it will create the stratums of different different categories
#so whatever observation you want depend on the predefined observations it will take the equal proportions from all the stratums

from sklearn.model_selection import train_test_split

#Select categorical variable
y = marketing_train['profession']
y

# Select Subset using stratified Sampling
# split is used to do the stratified sampling
# 0.6 means i want 60% of whole data to b stratified
Rest, Sample = train_test_split(marketing_train, test_size = 0.6, stratify = y)
Sample.head(10)


