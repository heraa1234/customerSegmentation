#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[3]:


data=pd.read_csv('Mall_Customers.csv')
data.head()


# In[4]:


data.drop("CustomerID",axis=1)


# In[5]:


data.isnull().sum()


# In[6]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
print(data_scaled)


# In[7]:


sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()


# In[8]:


# Convert categorical 'Gender' column to numerical
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})
data.head()


# In[9]:


plt.figure(figsize=(8,6))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[10]:


inertia=[]
K = range(1,11)
for k in K:
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)


# In[11]:


plt.figure(figsize=(8,6))
plt.plot(K,inertia,'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("Inertia")
plt.title("Elbow method to determine optimal K")
plt.show()


# In[12]:


optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

data['Cluster'] = clusters


# In[13]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='bright', data=data)
plt.title('Customer Segmentation Based on Clusters')
plt.show()


# In[14]:


cluster_profile = data.groupby('Cluster').mean()
print(cluster_profile)


# In[15]:


silhouette_avg = silhouette_score(data_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg:.3f}')


# In[16]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title('3D Visualization of Clusters')
plt.show()


# In[ ]:




