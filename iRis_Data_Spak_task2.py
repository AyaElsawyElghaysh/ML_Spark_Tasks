#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt


# # IMPORT IRIS DATAEST

# In[15]:


path="E:\\ITIAI2021\\Spark\\iris.csv"
iris_data=pd.read_csv(path)
iris_data


# In[16]:


iris_data.info()


# In[17]:


df=iris_data.drop(columns=["Id"])


# In[18]:


iris_data.describe()


# # try to see the realtion between each features by using pairplot 

# ### using seaborn  to show the relation  between features

# In[19]:


import seaborn as sns
sns.pairplot(df, hue="Species", markers=["o", "s", "D"])


# ## from the pair plot that setosa can be clustered easy beacause it is separated on vercicolor and virinca 
# and we notice that the most important feature that we can use it to cluster the data is petal_length and petal_width

# # Now i will use k-means cluster to get optimum number of classes

# In[20]:


from sklearn.cluster import KMeans


# In[40]:


#x = iris_data.iloc[:, [0, 1, 2, 3]].values
x = iris_data.iloc[:, [0, 1, 2, 3]].values
x


# In[41]:


wcss = []
for i in range(1, 11):    #11 number for clusters 
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# # from the elbow digram we choose **3** cluster

# In[42]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[43]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1],
            s = 100, c = 'green', label = 'Iris-virginica')
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:






# In[ ]:




