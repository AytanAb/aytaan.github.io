#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_csv("data_supervised.csv")


# In[3]:


data


# In[4]:


data.dtypes


# In[5]:


data.describe()


# In[6]:


data.isna().sum()/len(data)


# In[7]:


X =data['Hours']
y=data['Scores']


# In[8]:


import matplotlib.pyplot as plt
plt.scatter(data['Hours'], data['Scores'])
plt.title("Dependence between study hours and scores",fontsize = 14,color = 'green',weight = 'bold')
plt.xlabel("Scores",fontsize = 12,weight = 'bold')
plt.ylabel("Hours", fontsize =12, weight = 'bold')
plt.show()


# In[9]:


X = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state = 0)


# In[11]:


import numpy as np
model = LinearRegression()
lg = model.fit(X_train,y_train)


# In[12]:


import matplotlib.pyplot as plt
line = model.coef_*X + model.intercept_
plt.scatter(X, y)
plt.plot(X,line,color = 'red')
plt.title("Dependence between study hours and scores",fontsize = 14,color = 'green',weight = 'bold')
plt.xlabel("Scores",fontsize = 12,weight = 'bold')
plt.ylabel("Hours", fontsize =12, weight = 'bold')
plt.show()


# In[13]:


prediction = model.predict(X_test)


# In[14]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})  
df


# In[15]:


hours = [[9.25]]
own_pred = model.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[16]:


from sklearn import metrics  


# In[17]:


print(metrics.mean_absolute_error(y_test,prediction))


# In[ ]:




