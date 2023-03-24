#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation : Data Science and Business  Analytics Tasks

# ## Task 4 : Exploratory Data Analysis - Terrorism

# ### Author : Aytan Abdullayeva 

# In[ ]:


#import required libraries 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("assignment4.csv",encoding= 'latin1')
data


# In[ ]:


df = data.copy()


# In[4]:


for i in df.columns:
    print(i,end=", ")


# In[5]:


df = df[['iyear','imonth',"iday",'country_txt','region_txt','provstate','city','latitude','longitude','location','summary','attacktype1_txt','targtype1_txt','gname','motive','weaptype1_txt','nkill','nwound','addnotes']]


# In[6]:


df.head()


# In[7]:


df.rename(columns = {"iyear": "Year","imonth":"Month","iday":"Day",
                     "country_txt":"Country",'region_txt':'Region','provstate':'Province/State',
                     'city':"City",'latitude':"Latitude",'longitude':"Longitude",
                     'location':"Location",'summary':"Summary",'attacktype1_txt':"Attack Type",
                     'targtype1_txt':"Target Type",'gname': "Group Name",'motive':"Motive",
                     'weaptype1_txt':"Weapon Type",'nkill':"Killed",'nwound':"Wounded",
                     'addnotes':"Add Notes"},inplace=True)


# In[8]:


df


# In[9]:


df.isnull().sum()


# In[10]:


df['Killed'] = df['Killed'].fillna(0)
df["Wounded"] = df["Wounded"].fillna(0)
df["Casuality"] = df["Killed"] + df["Wounded"]


# In[11]:


df.describe()


# The data describes information about terrorism  between 1970 and 2017.
# The maximum latitude is approximately 75 and minimum is -53.
# The maximum number of people killed in terror is 1570.
# The maximum number of wounded people in terror is 8191.
# 

# # Data Visualizition

# In[12]:


df1=df.copy()


# In[13]:


df1 = df1.groupby("Year")["Year"].count()


# In[14]:


df1 = pd.DataFrame(df1)


# In[15]:


df1 = df1.rename(columns={"Year": "Attacks"})


# In[16]:


df1=df1.reset_index()


# In[17]:


df1.head()


# In[18]:


plt.bar(df1["Year"],df1["Attacks"],color="lightgreen")
plt.title("Attacks between 1970 and 2017",fontsize=16,weight="bold",color="green")
plt.xlabel("Years",fontsize=10,weight ="bold",color="green")
plt.ylabel("Number of Attacks",fontsize=10,weight="bold",color="green")


# In[19]:


df2=df.copy()
df2= df2.groupby("Year")["Casuality"].sum()
df2 = pd.DataFrame(df2)
df2 = df2.reset_index()


# In[20]:


df2.head()


# In[21]:


plt.bar(df1["Year"],df2["Casuality"],color="lightgreen")
plt.title("Attacks between 1970 and 2017",fontsize=16,weight="bold",color="green")
plt.xlabel("Years",fontsize=10,weight ="bold",color="green")
plt.ylabel("Number of Casuality",fontsize=10,weight="bold",color="green")


# In[22]:


df3=df.copy()
df3= df3.groupby("Year")["Killed"].sum()
df3 = pd.DataFrame(df3)
df3 = df3.reset_index()


# In[23]:


df3.head()


# In[24]:


df4=df.copy()
df4= df4.groupby("Year")["Wounded"].sum()
df4 = pd.DataFrame(df4)
df4 = df4.reset_index()


# In[25]:


df4.head()


# In[26]:


plt.bar(df1["Year"],df3["Killed"],color="lightgreen")
plt.title("The number of people who killed in between 1970 and 2017",fontsize=16,weight="bold",color="green")
plt.xlabel("Years",fontsize=10,weight ="bold",color="green")
plt.ylabel("Number of Killed people",fontsize=10,weight="bold",color="green")

plt.figure()
plt.bar(df1["Year"],df4["Wounded"],color="lightgreen")
plt.title("The number of people who wounded in between 1970 and 2017",fontsize=16,weight="bold",color="green")
plt.xlabel("Years",fontsize=10,weight ="bold",color="green")
plt.ylabel("Number of Wounded people",fontsize=10,weight="bold",color="green")


# # Visualization by Region

# In[27]:


region = pd.crosstab(df.Year,df.Region)


# In[28]:


region.head()


# In[29]:


region.plot(kind="area",figsize=(10,5))
plt.title("Region Attacks",fontsize=16,weight="bold",color="green")
plt.xlabel("Years",fontsize=10,color="green")
plt.ylabel("Number of Attacks",fontsize=10,color="green")
plt.show()


# In[30]:


region2 =region.transpose()
region2["Total Attacks"]=region2.sum(axis=1)


# In[31]:


reg=region2["Total Attacks"].sort_values(ascending=False)


# In[32]:


reg.plot(kind='bar',figsize=(10,5))
plt.title("The number of attacks in each region",fontsize=16,weight="bold",color="green")
plt.xlabel("Region",fontsize=10,color="green")
plt.ylabel("Total Attacks",fontsize=10,color ="green")
plt.show()


# In[33]:


df5=df.copy()
df5= df5.groupby("Region")["Casuality"].sum()
df5 = pd.DataFrame(df5)


# In[34]:


df5


# In[35]:


df5.plot(kind='bar',figsize=(10,5))
plt.title("The number of casualities in each region",fontsize=16,weight="bold",color="green")
plt.xlabel("Region",fontsize=10,color="green")
plt.ylabel("Total Casualities",fontsize=10,color ="green")
plt.show()


# In[36]:


df6=df.copy()
df6= df6.groupby("Region")["Killed"].sum()
df6 = pd.DataFrame(df6)

df7=df.copy()
df7= df7.groupby("Region")["Wounded"].sum()
df7 = pd.DataFrame(df7)


# In[37]:


df6.plot(kind='bar',figsize=(10,5))
plt.title("The number of killed people in each region",fontsize=16,weight="bold",color="green")
plt.xlabel("Region",fontsize=10,color="green")
plt.ylabel("Total killed people",fontsize=10,color ="green")
plt.show()

plt.figure()

df7.plot(kind='bar',figsize=(10,5))
plt.title("The number of wounded people in each region",fontsize=16,weight="bold",color="green")
plt.xlabel("Region",fontsize=10,color="green")
plt.ylabel("Total wounded people",fontsize=10,color ="green")
plt.show()


# In[38]:


df7.plot(kind='bar',figsize=(10,5))
plt.title("The number of wounded people in each region",fontsize=16,weight="bold",color="green")
plt.xlabel("Region",fontsize=10,color="green")
plt.ylabel("Total wounded people",fontsize=10,color ="green")
plt.show()


# # Visualization by Country

# In[39]:


country = pd.crosstab(df.Year,df.Country)


# In[40]:


country


# In[41]:


country.plot(kind="area",figsize=(10,5),legend=None)
plt.title("Country Attacks",fontsize=16,weight="bold",color="green")
plt.xlabel("Years",fontsize=10,color="green")
plt.ylabel("Number of Attacks",fontsize=10,color="green")
plt.show()


# In[42]:


country2 = country.transpose()


# In[43]:


country2["TotalAttacks"] = country2.sum(axis=1)


# In[44]:


country2


# ## Country Attacks - Top 10  

# In[50]:


coun=country2["TotalAttacks"].sort_values(ascending=False).head(10)
coun


# In[58]:


coun.plot(kind='bar',figsize=(10,5))
plt.title("The number of attacks in top 10 countries",fontsize=16,weight="bold",color="green")
plt.xlabel("Country",fontsize=10,color="green")
plt.ylabel("Total Attacks",fontsize=10,color ="green")
plt.show()


# In[56]:


df8=df.copy()
df8= df8.groupby("Country")["Casuality"].sum().sort_values(ascending=False).head(10)
df8 = pd.DataFrame(df8)


# In[57]:


df8


# In[59]:


df8.plot(kind='bar',figsize=(10,5))
plt.title("The casualities",fontsize=16,weight="bold",color="green")
plt.xlabel("Country",fontsize=10,color="green")
plt.ylabel("Total number of Casualities",fontsize=10,color ="green")
plt.show()


# In[63]:


df9=df.copy()
df9= df9.groupby("Country")["Killed"].sum().sort_values(ascending=False).head(10)
df9 = pd.DataFrame(df9)

df10=df.copy()
df10= df10.groupby("Country")["Wounded"].sum().sort_values(ascending=False).head(10)
df10 = pd.DataFrame(df10)


# In[67]:


df9.plot(kind='bar',figsize=(10,5))
plt.title("The number of killed people ",fontsize=16,weight="bold",color="green")
plt.xlabel("Country",fontsize=10,color="green")
plt.ylabel("The number of killed people",fontsize=10,color ="green")
plt.show()

plt.figure()

df10.plot(kind='bar',figsize=(10,5))
plt.title("The number of wounded people",fontsize=16,weight="bold",color="green")
plt.xlabel("Country",fontsize=10,color="green")
plt.ylabel("The number of wounded people",fontsize=10,color ="green")
plt.show()


# ## Visualization by Target type Attacks 

# In[68]:


target = df["Target Type"].value_counts()


# In[69]:


target


# In[89]:


import matplotlib.pyplot as plt

# create a figure with a grid of 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# plot the pie chart in the first subplot
target.plot(kind='pie', ax=ax1, labels=None)

# add the legend in the second subplot
ax2.axis('off')
ax2.legend(target.index, prop={"size": 6}, loc='center')
ax2.set_title('Legend')

# use bbox_to_anchor to position the legend next to the pie chart
ax1.legend(target.index, prop={"size": 6}, loc='center right', bbox_to_anchor=(1.5, 0.5))

plt.show()


# In[100]:


df11=df.copy()
df11= df11.groupby("Target Type")["Casuality"].sum().sort_values(ascending=False)
df11 = pd.DataFrame(df11)

df12=df.copy()
df12= df12.groupby("Target Type")["Killed"].sum().sort_values(ascending=False)
df12 = pd.DataFrame(df12)

df13=df.copy()
df13= df13.groupby("Target Type")["Wounded"].sum().sort_values(ascending=False)
df13 = pd.DataFrame(df13)


# In[102]:


df11.plot(kind='barh',figsize=(10,5))
plt.title("The Casualities ",fontsize=16,weight="bold",color="green")
plt.xlabel("The number of people",fontsize=10,color="green")
plt.ylabel("Country",fontsize=10,color ="green")
plt.show()

plt.figure()

df12.plot(kind='barh',figsize=(10,5))
plt.title("The Killed people ",fontsize=16,weight="bold",color="green")
plt.xlabel("The number of people",fontsize=10,color="green")
plt.ylabel("Country",fontsize=10,color ="green")
plt.show()

plt.figure()

df13.plot(kind='barh',figsize=(10,5))
plt.title("The Wounded people ",fontsize=16,weight="bold",color="green")
plt.xlabel("The number of people",fontsize=10,color="green")
plt.ylabel("Country",fontsize=10,color ="green")
plt.show()


# # Observations
1.Annual Attacks 
  1.a)The max number of attacks 16903 in 2014
    b)The min number of attacks 471 in 1971
    
  2.a)The max number of casualities 85618 in 2014
    b)The min number of casualities 255 in 1971
    
  3.a)The max number of killed people 44490 in 2014
    b)The min number of killed people 173 in 1971
   
  4.a)The max number of wounded people 44043 in 2015
    b)The min number of wounded people 82 in 1971
    
2.Regional Attacks 
  1.a)The max number of attacks 50474 in Middle East & North Africa
    b)The min number of attacks 282 in Australasia & Oceania
    
  2.a)The max number of casualities 351950 in Middle East & North Africa
    b)The min number of casualities 410 in Australasia & Oceania
    
  3.a)The max number of killed people 137642 in Middle East & North Africa
    b)The min number of killed people 150 in Australasia & Oceania
   
  4.a)The max number of wounded people 214308 in Middle East & North Africa
    b)The min number of wounded people 260 in Australasia & Oceania

3.Top 10 Attacks by Country
  1.a)The max number of attacks 7589 in Baghdad
    b)The min number of attacks 1019 in Athens
    
  2.a)The max number of casualities 77876 in Baghdad
    b)The min number of casualities 5748 in Aleppo
    
  3.a)The max number of killed people 21151 in Baghdad
    b)The min number of killed people 2125 in Aleppo
   
  4.a)The max number of wounded people 56725 in Baghdad
    b)The min number of wounded people 4955 in Mogadishu
    
4.Target Type Attacks
  1.a)The max number of attacks 43511 over Private Citizens &Property
    b)The min number of attacks 263 over Abortion Related 
    
  2.a)The max number of casualities 319176 over Private Citizens &Property
    b)The min number of casualities 56 over Abortion Related
    
  3.a)The max number of killed people 140504 in over Private Citizens &Property
    b)The min number of killed people 10 over Abortion Related
   
  4.a)The max number of wounded people 178672 over Private Citizens &Property
    b)The min number of wounded people 46 over Abortion Related
  
# In[ ]:




