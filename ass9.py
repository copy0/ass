#!/usr/bin/env python
# coding: utf-8

# # Assignment 9
# ## Data Visualization II
# ### 1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution of age with respect to each gender along with the information about whether they survived or not. (Column names : 'sex' and 'age')
# ### 2. Write observations on the inference from the above statistics.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


df = sns.load_dataset('titanic')


# In[3]:


df.isnull().sum()


# In[4]:


df['age'].fillna(df['age'].mean(), inplace = True)


# In[5]:


df.isnull().sum()


# In[6]:


df['embarked'].value_counts()


# In[7]:


df['embarked'].fillna('S', inplace = True)


# In[8]:


df.isnull().sum()


# In[9]:


df.fillna(method = 'ffill', inplace = True)


# In[10]:


df.isnull().sum()


# In[11]:


df.fillna(method = 'bfill', inplace = True)


# In[12]:


df.isnull().sum()


# In[13]:


sns.boxplot(x = 'sex', y = 'age', hue = 'survived', data = df)

