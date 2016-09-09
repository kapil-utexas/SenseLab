
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

dataset = np.loadtxt("data.csv", delimiter=",")


# In[3]:

dataset


# In[4]:

import matplotlib.pyplot as plt


# In[5]:

get_ipython().magic(u'matplotlib qt')


# In[6]:

plt.scatter(dataset[:,0],dataset[:,1], c='blue', s=10)


# In[ ]:

plt.show(block=True)


# In[ ]:



