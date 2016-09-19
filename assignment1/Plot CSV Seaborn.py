
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

import seaborn as sns


# In[3]:

data = pd.read_csv("q1_data.csv")


# In[4]:

features_to_plot = ['x','y']


# In[5]:

df_to_plot = data.ix[::,features_to_plot]


# In[6]:

df_to_plot


# In[7]:

get_ipython().magic(u'matplotlib')


# In[8]:

sns.pairplot(df_to_plot, hue="c")
plt.show(block = True)

# In[ ]:



