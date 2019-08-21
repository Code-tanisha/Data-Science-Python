import os
import pandas as pd
import numpy as np
os.chdir("C:\Python")
marketing_train = pd.read_csv("marketing_tr.csv")
marketing_train.head()
df = marketing_train.copy()
df

df.dtypes
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()
obj_df["schooling"].value_counts()
# We need to clean null values
obj_df[obj_df.isnull().any(axis=1)]
obj_df["schooling"].value_counts()
# we will replace null values. so after this data will not be having any null values
obj_df = obj_df.fillna({"schooling" : "four"})
obj_df

# So fromhere data does not have any null values so we can look at options for encoding
obj_df["marital"].value_counts()

cleanup_nums = {"marital": {"married": 1, "single": 2, "divorced": 0, "unknown": 8},
               "schooling": {"four": 4, "university.degree": 1, "high.school": 2, "basic.9y": 3, "professional.course": 5, "basic.4y": 6, "basic.6y": 7, "unknown": 8, "illiterate": 9},
               "profession": {"admin.": 0, "blue-collar": 1, "technician": 2, "services": 3, "management": 4, "retired": 5, "entrepreneur": 6, "self-employed": 7, "housemaid": 8, "unemployed": 9, "student" : 10, "unknown": 11},
               "housing": {"no": 0, "yes": 1, "unknown": 2}, "loan": {"no": 0, "yes": 1, "unknown": 2}, "responded": {"no": 0, "yes": 1},
               "contact": {"cellular": 0, "telephone": 1}, "month": {"may": 0, "jul": 1, "aug": 2, "jan": 3, "nov": 4, "apr": 5, "oct": 6, "sep": 7, "mar": 8, "dec": 9},
               "day_of_week": {"mon": 0, "thu": 1, "tue": 2, "wed": 3, "fri": 4}, "poutcome": {"nonexistent": 0, "failure": 1, "success": 2},
               "default": {"no": 0, "unknown": 2, "yes": 1}}
cleanup_nums

# to convert the columns to numbers using replace
obj_df.replace(cleanup_nums, inplace=True)
obj_df.head()

obj_df.dtypes
