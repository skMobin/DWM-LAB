#K-means Clustering Algo

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

#Using the elbow method to find the  optimal number of clusters

from sklearn.cluster import KMeans
wcss = []   # within Cluster Sum Of Square
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', max_iter=300,n_init=10 , random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title("The Elbow method")
plt.xlabel('Number Of Cluster')
plt.ylabel('WCSS')
plt.show()

#applying kmeans to all dataset
 kmeans = KMeans(n_clusters= 5, init='k-means++', max_iter=300,n_init=10 , random_state=0)
 
y_kmeans = kmeans.fit_predict(x)
 
#visualizing the Cluster
plt.scatter(x[y_kmeans == 0,0],
            x[y_kmeans == 0,1],s =100, c='red',
            label = 'Careful')

plt.scatter(x[y_kmeans == 1,0],
            x[y_kmeans == 1,1],s =100, c='blue',
            label = 'Standard')

plt.scatter(x[y_kmeans == 2,0],
            x[y_kmeans == 2,1],s =100, c='green',
            label = 'Target')

plt.scatter(x[y_kmeans == 3,0],
            x[y_kmeans == 3,1],s =100, c='cyan',
            label = 'Careless')

plt.scatter(x[y_kmeans == 4,0],
            x[y_kmeans == 4,1],s =100, c='magenta',
            label = 'Sensible')

plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1], s = 300,
            c = 'yellow', label = 'Centriods')
plt.title('Clusters Of clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending SCore (1-100)')
plt.legend()
plt.show()

 