#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.cluster import KMeans 


# In[25]:


data =datasets.load_iris()
df =pd.DataFrame(data.data, columns = data.feature_names)


# In[26]:


df


# In[27]:


df.dropna(inplace=True)


# In[28]:


df.isnull().sum()/len(data)


# In[49]:


x = df.iloc[: , :].values #convert Dataframe to numpy array 
arr = []

for i in range(1,31):
    model = KMeans(n_clusters = i, init = 'k-means++', max_iter =200,n_init =15,random_state =0)
    model.fit(x)
    arr.append(model.inertia_)
print(arr)


# In[50]:


plt.plot(range(1,31),arr,color ="green")
plt.title("The Elbow method",fontsize =15 ,color ="red",weight = "bold")
plt.xlabel("Number of Clusters")
plt.show()


# The optimal number of clusters is 3

# In[51]:


model = KMeans(n_clusters = 3, init = 'k-means++', max_iter =200,n_init =15,random_state =0)
y= model.fit_predict(x)


# In[52]:


y


# In[55]:


plt.scatter(x[y == 0, 0], x[y == 0, 1], 
            s = 100, c = 'green', label = 'Iris-setosa')
plt.scatter(x[y == 1, 0], x[y == 1, 1], 
            s = 100, c = 'purple', label = 'Iris-versicolour')
plt.scatter(x[y == 2, 0], x[y == 2, 1],
            s = 100, c = 'yellow', label = 'Iris-virginica')

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], 
            s = 100, c = 'black', label = 'Centroids')

plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




