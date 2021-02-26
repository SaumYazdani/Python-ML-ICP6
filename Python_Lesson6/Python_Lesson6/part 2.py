import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
#read data
dataset = pd.read_csv('data')
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:,-1]
#set all null values in columns to average of that column
dataset['CREDIT_LIMIT'].fillna((dataset['CREDIT_LIMIT'].mean()), inplace=True)
dataset['MINIMUM_PAYMENTS'].fillna((dataset['MINIMUM_PAYMENTS'].mean()), inplace=True)
#fitting data to be scaled
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)
#setting clusters to 2
from sklearn.cluster import KMeans
nclusters = 2
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
# elbow method to know the number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
#plotting resutlts
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
#calculating silhouette score
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)
#silhoutte score is slighlty lower that part 1's score. This means the clusters are closer to neighboring clusters.
#This is because there is less distinction between values and thus, similar values have a higher tendancy to overlap
#and have less of a tight grouping.