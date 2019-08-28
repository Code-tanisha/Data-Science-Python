import os
import pandas as pd
import numpy as np

 # os.chdir("C:\Python")
 # data = pd.read_csv("iris.csv")
# data

raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
           'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
           'age': [42, 52, 36, 24, 73],
           'preTestscore': [4, 24, 31, 2, 3],
           'postTestscore': [25, 94, 57, 62, 70]}
data = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestscore', 'postTestscore'])
data

# Select a random subset of 2 without replacement
data.take(np.random.permutation(len(data))[:2])

# Select a random subset of 4 without replacement
data.take(np.random.permutation(len(data))[:4])

# random subset of data
df1 = data.sample(3)
df1
