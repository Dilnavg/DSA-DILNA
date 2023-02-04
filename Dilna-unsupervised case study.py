#!/usr/bin/env python
# coding: utf-8

# In[216]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[217]:


data=pd.read_csv(r'C:\DSA\Wine_clust.csv')


# In[218]:


data.head()


# In[219]:


data.info()


# In[220]:


data.isna().sum()


# In[221]:


data.shape


# In[222]:


data.columns


# In[223]:


data.describe()


# In[224]:


data.nunique()


# In[225]:


k=1
plt.figure(figsize=(10,10))
plt.suptitle("Distribution of Outliers")

for i in data:
    plt.subplot(5,3,k)
    sns.boxplot(x = i, data = data)
    plt.title(i)
    plt.tight_layout()
    k+=1


# In[226]:


##few outliers 


# In[227]:


data.hist(figsize=(12,12));


# # K MEANS CLUSTERING

# In[228]:


from sklearn.cluster import KMeans


# In[229]:


wcss=[]
for i in range(1,13):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,13),wcss)
plt.title('The Elbow Method')
plt.xlabel('No.of Clusters')
plt.ylabel('WCSS value')
plt.show()


# In[230]:


from yellowbrick.cluster import KElbowVisualizer


# In[231]:


from yellowbrick.cluster import KElbowVisualizer


# In[232]:


kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans,k=(1,13))
visualizer.fit(data)
visualizer.poof()
plt.show()


# In[233]:


#optimum number of clusters=3 it can be visualize through yellowbrick


# # final model

# In[234]:


#clustering is done. to find how many elements each cluster consists of


# In[235]:


kmeans=KMeans(n_clusters=3,init="k-means++").fit(data)


# In[236]:


cluster=kmeans.labels_


# In[237]:


data["cluster_no"]=cluster


# In[238]:


data.cluster_no.value_counts()


# In[239]:


data.head()


# In[240]:


data1=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values


# In[241]:


data1


# In[242]:


type(data1)


# In[243]:


kmeans=KMeans(n_clusters=3,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(data1)


# In[244]:


y_kmeans


# In[245]:


plt.scatter(data1[y_kmeans==0,0],data1[y_kmeans==0,12],s=100,c='blue',label='Cluster 1')
plt.scatter(data1[y_kmeans==1,0],data1[y_kmeans==1,12],s=100,c='red',label='Cluster 2')
plt.scatter(data1[y_kmeans==2,0],data1[y_kmeans==2,12],s=100,c='green',label='Cluster 3')
plt.title('Kmeans Clustering plot for Wine dataset')
plt.legend()
plt.show()


# In[ ]:





# # Hierarchical Clustering

# In[246]:


df=pd.read_csv(r'C:\DSA\Wine_clust.csv')


# In[247]:


df.head()


# In[248]:


df.columns


# In[249]:


df.isna().sum()


# In[250]:


df.shape


# In[251]:


import scipy.cluster.hierarchy as sch


# In[252]:


dendrogram=sch.dendrogram(sch.linkage(df,method='ward'))
plt.title('Dndrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()


# In[253]:


from sklearn.cluster import AgglomerativeClustering


# In[254]:


ahc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_ahc=ahc.fit_predict(df)


# In[255]:


y_ahc


# In[256]:


from sklearn.metrics import silhouette_score
sil_ahc=silhouette_score(df,y_ahc)


# In[257]:


sil_ahc


# In[258]:


ahc1=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
y_ahc1=ahc1.fit_predict(df)


# In[259]:


from sklearn.metrics import silhouette_score
sil_ahc1=silhouette_score(df,y_ahc1)


# In[260]:


sil_ahc1


# In[261]:


df1=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values


# In[262]:


type(df1)


# In[263]:


cluster_H = AgglomerativeClustering(n_clusters=3)
model_clt = cluster_H.fit(df1)
model_clt
pred1 = model_clt.labels_
pred1


# In[264]:


plt.scatter(df1[pred1 == 0, 9], df1[pred1 == 0, 4], s = 20, c = 'orange', label = 'Type 0')
plt.scatter(df1[pred1 == 1, 3], df1[pred1 == 1, 12], s = 20, c = 'yellow', label = 'Type 1')
plt.scatter(df1[pred1 == 2, 1], df1[pred1 == 2, 12], s = 20, c = 'green', label = 'Type 2')
plt.title('Hierarchical Plot for Wine dataset')
plt.legend()


# In[265]:


## minimum number of cluster =2 gives the best result in hierarchical clustering


# In[266]:


cluster_H1 = AgglomerativeClustering(n_clusters=2)
model_clt1 = cluster_H1.fit(df1)
model_clt1
pred2 = model_clt1.labels_
pred2


# In[267]:


plt.scatter(df1[pred2 == 0, 9], df1[pred2 == 0, 4], s = 20, c = 'orange', label = 'Type 0')
plt.scatter(df1[pred2 == 1, 3], df1[pred2 == 1, 12], s = 20, c = 'yellow', label = 'Type 1')
plt.title('Hierarchical Plot for Wine dataset')
plt.legend()


# # DBSCAN algorithm

# In[ ]:


#DBSCAN is abbreviation for Density-Based Spatial Clustering of Application with Noise algorithm. It is a method of clustering by separate high-density points from low-density points.


# In[373]:


df=pd.read_csv(r'C:\DSA\Wine_clust.csv')


# In[374]:


df.head()


# In[375]:


#x=df.iloc[:,[5,6]].values


# In[376]:


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(df)
distances, indices = nbrs.kneighbors(df)


# In[377]:


import matplotlib.pyplot as plt
distances = np.sort(distances, axis=0)
distances = distances[:,4]
plt.plot(distances)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


# In[378]:


from sklearn.cluster import DBSCAN
db=DBSCAN(eps=75,min_samples=5,metric='euclidean')


# In[379]:


model=db.fit(df)


# In[380]:


label=model.labels_


# In[381]:


label


# In[382]:


from sklearn import metrics

#identifying the points which makes up our core points
sample_cores=np.zeros_like(label,dtype=bool)

sample_cores[db.core_sample_indices_]=True

#Calculating the number of clusters

n_clusters=len(set(label))- (1 if -1 in label else 0)
print('No of clusters:',n_clusters)


# In[383]:


y_means = db.fit_predict(x)
plt.figure(figsize=(7,5))
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 50, c = 'pink')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 50, c = 'red')

plt.title('DB scan Plot for Wine dataset')
plt.legend()


# In[340]:





# In[ ]:





# In[342]:





# In[ ]:





# In[344]:





# In[349]:





# In[350]:





# In[ ]:





# In[ ]:





# In[ ]:




