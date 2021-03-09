# -*- coding: utf-8 -*-
! pip install scikit-plot

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

import scikitplot as skplot

"""# Data Cleaning"""

# reading the csv 
cc_org = pd.read_csv("CC GENERAL.csv")
cc_org.shape

# first look at the data set
cc_org.head()

# looking at the data set and the data type 
cc_org.info()

# looking for missing value 
# detected missing values for "minimum payments" and "credit_limit"
cc_org.isna().sum()

# looking at general numbers like min, max, average for every feature to evaluate how to deal with missing values
cc_org.describe().T

# drop "cust_id" since it doesn't have any role in determining the cluster and is not numeric
cc_org = cc_org.drop(columns = "CUST_ID")

# change all missing value under "minimum payments" to median
# we chose the median because the mean was significantly higher and seemed not appropriate
cc_org['MINIMUM_PAYMENTS'].fillna(cc_org['MINIMUM_PAYMENTS'].median(), inplace=True)

# Just 1 value is missing at CREDIT_LIMIT feature
cc_clean = cc_org.dropna()

# lower case all column names
cc_clean.columns = cc_clean.columns.str.lower()

# verify that we don't have any missing values left
cc_clean.isna().sum()

# check for duplicates
cc_clean[cc_clean.duplicated()].count()

# no dupplicates

"""# EDA"""

# correlations
cc_corr = cc_clean.corr()
sns.heatmap(cc_corr)

# plot all the features
# most of them are skewed right 
plt.figure(figsize=(20,35))
for i, col in enumerate(cc_clean):
  if cc_clean[col].dtype != 'object' :
    ax = plt.subplot(9,2, i +1)
    sns.distplot(cc_clean[col], ax = ax, color='#fdc029')
    plt.xlabel(col)
    
plt.show()

# finding outliers of balance
sns.boxplot(cc_clean['balance'], color='#fdc029')
plt.show()

# finding outliers of purchases
sns.boxplot(cc_clean['purchases'], color='#fdc029')
plt.show()

# finding outliers of cash_advance
sns.boxplot(cc_clean['cash_advance'], color='#fdc029')
plt.show()

# finding outliers of payments
sns.boxplot(cc_clean['payments'], color='#fdc029')
plt.show()

# plotting balance and different purchase options 
plt.figure(figsize=(10,4))
sns.lineplot(cc_clean['balance'],cc_clean['purchases'],label="Purchases")
sns.lineplot(cc_clean['balance'],cc_clean['oneoff_purchases'],label='Oneoff')
sns.lineplot(cc_clean['balance'],cc_clean['installments_purchases'],label='Install')
plt.show()

sns.pairplot(cc_clean)

"""# Hierarchical clustering"""

# using standardscaler to reduce the distance of each variable
sc = StandardScaler()
xs = sc.fit_transform(cc_clean)
X = pd.DataFrame(xs, index=cc_clean.index, columns=cc_clean.columns)

# hclust
# going to do euclidean, cosine and cityblock distance
# no jaccard since this is more for categorical problems
diste = pdist(X.values)
distc = pdist(X.values, metric='cosine')
distm = pdist(X.values, metric ='cityblock')


hclust_e = linkage(diste)
hclust_c = linkage(distc)
hclust_m = linkage(distm)

# setting a higher limit to be able to generate the dendrogram
import sys
sys.setrecursionlimit(10000)

# both plot
LINKS = [hclust_e, hclust_c, hclust_m]
TITLE = ['Euclidean','Cosine','Manhattan']

plt.figure(figsize=(15,5))

# loop and build our plot
for i, m in enumerate(LINKS):
  plt.subplot(1,3,i+1)
  plt.title(TITLE[i])
  dendrogram(m,
             leaf_rotation=90,
             orientation='left')
plt.show()

# picking cosine ans using 4 different methods to detect the clusters
METHODS = ['single', 'complete', 'average','ward']
plt.figure(figsize=(20,5))

for i, m in enumerate(METHODS):
  plt.subplot(1,4,i+1)
  plt.title(m)
  dendrogram(linkage(distc, method=m),
             leaf_rotation=90)
plt.show()

# cosine and average seems like a good combination, so let's take a closer look
plt.figure(figsize=(10, 6))

avg = linkage(X.values, method="average")
dendrogram(avg,
          labels = X.index,
          leaf_rotation=90,
          leaf_font_size=10, color_threshold=6)

plt.axhline(y=8)
plt.show()

# using cosine + complete
# the labels with 4 clusters
labs = fcluster(linkage(distc, method='average'), 6, criterion='maxclust')

# confirm
np.unique(labs)

# add a cluster column to the stocks data set
cc_clean['cluster'] = labs

# How many credit card per cluster assignment?
cc_clean.cluster.value_counts(dropna=False,sort=False)

"""# KMeans clustering"""

# another method: KMeans 
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(cc_clean)

# Kmeans for 2 to 8 clusters
KS = range(2, 10)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  km.fit(X_pca)
  labs = km.predict(X_pca)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(X_pca, labs))

print(silo)

# plot 
plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)

plt.show()

# first, lets get the explained variance
# elbow plot

varexp = pca.explained_variance_ratio_
sns.lineplot(range(1, len(varexp)+1), varexp)

# cumulative variance

plt.title("Cumulative Explained Variance")
plt.plot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.987)

# quick function to construct the barplot easily
def ev_plot(ev):
  y = list(ev)
  x = list(range(1,len(ev)+1))
  return x, y

x, y = ev_plot(pca.explained_variance_)
sns.barplot(x, y)

# the clusters
hc_labs = fcluster(avg, 6, criterion="distance")

# the metrics
hc_silo = silhouette_score(X, hc_labs)
hc_ssamps = silhouette_samples(X, hc_labs)
np.unique(hc_labs)

skplot.metrics.plot_silhouette(X, hc_labs, title="HClust", figsize=(15,5))
plt.show()

# get the model
k3 = KMeans(3)
k3_labs = k3.fit_predict(X)

# metrics
k3_silo = silhouette_score(X, k3_labs)
k3_ssamps = silhouette_samples(X, k3_labs)
np.unique(k3_labs)

skplot.metrics.plot_silhouette(X, k3_labs, title="KMeans - 3", figsize=(15,5))
plt.show()

"""KMeans with 3 Clusters gives us a better silhouette score and has less negative values

"""

cc_clean['k3_labs'] = k3_labs

# profile
profile = cc_clean.groupby('k3_labs').mean()
profile

# heatmap 
sc = StandardScaler()
profile_scaled = sc.fit_transform(profile)

plt.figure(figsize=(12, 6))
pal = sns.color_palette("vlag", as_cmap=True)
sns.heatmap(profile_scaled, center=0, cmap=pal, xticklabels=profile.columns)

