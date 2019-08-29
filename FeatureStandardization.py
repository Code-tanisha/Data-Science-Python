# How to standardize features in python
import os
from sklearn import preprocessing
import numpy as np

# create feature
x = np.array([[-500.5], [-100.1], [0], [100.1], [900.9]])
x

# Standardize feature
# Create scaler
scaler = preprocessing.StandardScaler()
scaler

# Transform the feature
standardized_x = scaler.fit_transform(x)
standardized_x
