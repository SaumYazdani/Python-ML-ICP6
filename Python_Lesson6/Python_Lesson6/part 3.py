import pandas as pd
from sklearn.decomposition import PCA
#reading data and setting null values to the column average
dataset = pd.read_csv('data')
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:,-1]
dataset['CREDIT_LIMIT'].fillna((dataset['CREDIT_LIMIT'].mean()), inplace=True)
dataset['MINIMUM_PAYMENTS'].fillna((dataset['MINIMUM_PAYMENTS'].mean()), inplace=True)
#fitting data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)
# Apply transform to both the training set and the test set.
x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2,dataset[['TENURE']]],axis=1)
#calculating KMeans and fitting finaldf data
from sklearn.cluster import KMeans
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(finaldf)
y_cluster_kmeans = km.predict(finaldf)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(finaldf)
    wcss.append(kmeans.inertia_)
from sklearn import metrics
#caulculating silhouette score
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)
#silhoutte score is negative. This means there are clusters overlapping. Since the data is fairly similar,
#adding PCA and reducing dimensions creates more overlapping.