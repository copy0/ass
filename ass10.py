#!/usr/bin/env python
# coding: utf-8

# # Assignment 10
# ## Data Visualization III
# ### Download the Iris flower dataset or any other dataset into a DataFrame. (e.g., https://archive.ics.uci.edu/ml/datasets/Iris ). Scan the dataset and give the inference as:
# 
# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
# 3. Create a boxplot for each feature in the dataset.
# 4. Compare distributions and identify outliers.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


df = sns.load_dataset('iris')


# ### 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.

# In[4]:


df.dtypes


# ### 2. Create a histogram for each feature in the dataset to illustrate the feature distributions.

# In[5]:


sns.histplot(df['sepal_length'])


# In[6]:


sns.kdeplot(df['sepal_length'])


# In[7]:


sns.distplot(df['sepal_length'], kde = True,)


# In[8]:


df['sepal_length'].skew()


# In[9]:


sns.histplot(df['sepal_width'])


# In[10]:


sns.kdeplot(df['sepal_width'])


# In[11]:


sns.histplot(df['petal_length'])


# In[12]:


sns.kdeplot(df['petal_length'])


# In[13]:


sns.histplot(df['petal_width'])


# In[14]:


sns.histplot(df['species'])


# In[ ]:





# ### 3. Create a boxplot for each feature in the dataset.

# In[15]:


sns.boxplot(df['petal_length'])


# In[16]:


sns.boxplot(df['petal_width'])


# In[17]:


sns.boxplot(df['sepal_length'])


# In[18]:


sns.boxplot(df['sepal_width'])


# In[19]:


sns.boxplot(df)


# ### 4. Compare distributions and identify outliers.

# In[20]:


sns.kdeplot(df)

