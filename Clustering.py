import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

os.chdir("C:\Python")
df = pd.read_csv("iris.csv")
df

# Load required libraries
from sklearn.cluster import KMeans

# Estimate the optimum nos of clusters
cluster_range = range(1, 20)
cluster_errors = []

#loop will iteratively take the no of clusters to be built eg 1st iteration it will take 2 clusters and find the error and then three till 20 it will take
# kMeans concept is used to extract the method erroes and variance
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters).fit(df.iloc[:,0:4])
    cluster_errors.append(clusters.inertia_)
# inertia will give us the amount of errors which i am getting from each clusters

# Create a dataframe whith cluster errors ie. these cluster_errors list
clusters_df = pd.DataFrame({"num_clusters":cluster_range, "cluster_errors": cluster_errors})
clusters_df

# Plot line chart to vizualize number of clusters

%matplotlib inline
plt.figure(figsize=(12,6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o")
# x axis = no of clusters, y = errors

# df = data set and using the iloc we are selecting thw starting 4 variables
kmeans_model = KMeans(n_clusters = 3).fit(df.iloc[:,0:4])
kmeans_model.labels_

# Summarize output
# here we need to cheq the performance of the kmeans model
pd.crosstab(df['Species'], kmeans_model.labels_)


