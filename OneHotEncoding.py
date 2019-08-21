import os
import pandas as pd
import category_encoders as ce
os.chdir("C:\Python")
marketing_train = pd.read_csv("marketing_tr.csv")
df = marketing_train.copy()
df
df.dtypes

obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

#one hot encoding
#create an object of the one hot encoding
ce_OHE = ce.OneHotEncoder(cols=['profession', 'contact', 'responded'])

#tranform the data
df = ce_OHE.fit_transform(df)
df.head()
