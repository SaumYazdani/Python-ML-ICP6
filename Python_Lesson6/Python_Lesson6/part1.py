import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
#read data and find rows where null values exist
dataset = pd.read_csv('data')
print(dataset.isnull().sum())
#set all null values in columns to average of that column
dataset['CREDIT_LIMIT'].fillna((dataset['CREDIT_LIMIT'].mean()), inplace=True)
dataset['MINIMUM_PAYMENTS'].fillna((dataset['MINIMUM_PAYMENTS'].mean()), inplace=True)
print(dataset.isnull().sum())
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:,-1]
# see how many samples we have of each species
print(dataset["TENURE"].value_counts())
from sklearn.cluster import KMeans
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
# elbow method to know the number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#plotting
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
#finding silhouette score
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)
