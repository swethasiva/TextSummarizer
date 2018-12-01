#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.DataFrame({'x':[12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72], 'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]})


# In[3]:


# When you set a seed, the seed value is multiplied my a large number and then modulo of it is calculated and this becomes the random number selected and this is used as new seed to calculate next random numbers
# When we set a seed, the same calculations are repeated. So we get the same random numbers. This can be used for easier debugging
# By setting a seed value, Our code will always take in same random numbers and generate the same output.
np.random.seed(200) 

# Define the number of clusters
k=3

# Creating a Dictionary of Random Initial Centroids
centroids = {
            i+1 : [np.random.randint(0, 80), np.random.randint(0, 80)]
            for i in range(k)
        
}

# Plotting Data Points other than the chosen Centroids
plt.scatter(df['x'], df['y'], color='k')

# Color Mapping for the clusters
colmap = {1: 'r', 2:'g', 3:'b'}

# Plotting the Centroids with corresponding Cluster colors

for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()


# In[4]:


# Function to assign clusters to the dataitems based on eucledian distance from the previously chosen centroids
def clusterAssignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)]=(np.sqrt((df['x'] - centroids[i][0])**2 + (df['y'] - centroids[i][1])**2))
    
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x : colmap[x])
    return df

df = clusterAssignment(df, centroids)
print(df.head())

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()


# In[9]:


import copy

# Copy module has 2 types of copy:
# deepcopy(object) - Creates a new object which is an independent copy of the original object 
# copy(object) - Shallow Copy - creates a new object which references the original object
# deepcopy changes doesnt affect the original copy, shallowcopy affects original object values
old_centroids = copy.deepcopy(centroids)


# Function to recalibarate centroids based on the assignment step
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest']==i]['x'])
        centroids[i][1] = np.mean(df[df['closest']==i]['y'])
    return k
while True:
    closest_centroids = copy.deepcopy(df['closest'])
    centroids = update(centroids)
    df = clusterAssignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break


plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()


# In[ ]:





# In[ ]:




