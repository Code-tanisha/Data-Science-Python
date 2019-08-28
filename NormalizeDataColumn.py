import os
import pandas as pd
from sklearn import preprocessing 

 #os.chdir("C:\Python")
 #data = pd.read_csv("iris.csv")
#data.head()

df = {'score': [234,34,14,27,-74,160,-18,59]}
data = pd.DataFrame(df)
data

# Normalize the column
# Create x, where x is the score column calues as floats
x = data[['score']].values.astype(float)
x

# Create minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

min_max_scaler

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the DataFrame
data_normalized = pd.DataFrame(x_scaled)

data_normalized

