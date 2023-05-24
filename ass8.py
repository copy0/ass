#!/usr/bin/env python
# coding: utf-8

# # Assignment 08
# ## Data Visualization I
# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we can find any patterns in the data.
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by plotting a histogram.

# ### Importing libraries such as Numpy, Pandas, Seaborn and loading dataset titanic into our variable df

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


df = sns.load_dataset('titanic')


# ### Some of the basic operations performed on the dataset to get information about that dataset 

# In[4]:


df.shape


# In[5]:


df.head(10)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# ### Treating null values

# In[9]:


df['age'].fillna(df['age'].mean(), inplace = True)


# In[10]:


df.isnull().sum()


# In[11]:


df['embarked'].value_counts()


# In[12]:


df['embarked'].fillna('S', inplace = True)


# In[13]:


df.isnull().sum()


# In[14]:


df['deck'].fillna(method = 'ffill', inplace = True)


# In[15]:


df.isnull().sum()


# In[16]:


df['deck'].fillna(method = 'bfill', inplace = True)


# In[17]:


df.isnull().sum()


# In[18]:


df['embark_town'].value_counts()


# In[19]:


df['embark_town'].fillna('Southampton', inplace = True)


# In[20]:


df.isnull().sum()


# ### Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by plotting a histogram.

# In[21]:


plt.figure(figsize = (10, 7))
sns.histplot(df['fare'], bins = 10)


# ## EDA (Exploratory Data Analysis )
# ### Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we can find any patterns in the data.

# In[22]:


df.info()


# In[23]:


df['survived'].value_counts()


# In[24]:


df['pclass'].value_counts()


# In[25]:


df['sex'].value_counts()


# In[26]:


df['age'].value_counts()


# In[27]:


df['sibsp'].value_counts()


# In[28]:


df['parch'].value_counts()


# In[29]:


df['fare'].value_counts()


# In[30]:


df['embarked'].value_counts()


# In[31]:


df['who'].value_counts()


# In[32]:


plt.figure(figsize = (10, 7))
sns.histplot(df['age'], bins = 10)


# In[33]:


plt.figure(figsize = (10, 7))
sns.histplot(df['fare'], bins = 10)


# In[34]:


plt.figure(figsize = (10, 7))
sns.boxplot(df['age'])


# In[35]:


plt.figure(figsize = (10, 7))
sns.boxplot(df['fare'])


# In[36]:


plt.figure(figsize = (10, 7))
sns.distplot(df['age'])


# In[37]:


plt.figure(figsize = (10, 7))
sns.distplot(df['fare'])


# In[38]:


sns.kdeplot(df['age'])


# In[39]:


sns.kdeplot(df['fare'])


# In[40]:


df['age'].skew()


# In[41]:


df['fare'].skew()


# In[42]:


df[df['fare'] > 300]


# In[43]:


# Defininig function for Outliers Treatement

def Outlier_Treatment(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    upper = Q3 + (1.5 * IQR)
    lower = Q1 - (1.5 * IQR)
    np.clip(df[col], lower, upper, inplace = True)


# In[44]:


Outlier_Treatment('age')
plt.figure(figsize = (10, 7))
sns.boxplot(df['age'])


# In[45]:


Outlier_Treatment('fare')
plt.figure(figsize = (10, 7))
sns.boxplot(df['age'])


# In[46]:


df.info()


# ### We are using countplot for all columns except age, fare some examples:

# In[47]:


plt.figure(figsize = (10, 7))
sns.countplot(data = df, x = 'survived')


# In[48]:


plt.figure(figsize = (10, 7))
sns.countplot(data = df, x = 'pclass')


# In[49]:


plt.figure(figsize = (10, 7))
sns.countplot(data = df, x = 'sex')


# In[50]:


plt.figure(figsize = (10, 7))
sns.countplot(data = df, x = 'deck')


# ### We are using Piechart for all columns except age, fare some examples:

# In[51]:


df['deck'].value_counts().plot(kind = 'pie', autopct = '%.3f%%')


# In[52]:


df['survived'].value_counts().plot(kind = 'pie', autopct = '%.3f%%')


# In[53]:


df['embarked'].value_counts().plot(kind = 'pie', autopct = '%.3f%%')


# ### Bivariate Analysis

# In[54]:


sns.scatterplot(x = 'age', y = 'fare', data = df)
plt.show()


# In[55]:


sns.pairplot(data = df)
plt.show()


# In[56]:


sns.boxplot(x = 'survived', y = 'age', data = df)


# In[57]:


sns.barplot(x = 'survived', y = 'age', data = df)


# In[58]:


sns.barplot(x = 'survived', y = 'fare', data = df)


# In[59]:


sns.barplot(x = 'class', y = 'fare', data = df)


# In[60]:


sns.barplot(x = 'sex', y = 'survived', data = df)


# In[61]:


sns.heatmap(pd.crosstab(df['survived'], df['sex']), annot = True)


# In[62]:


sns.heatmap(pd.crosstab(df['survived'], df['class']), annot = True)


# In[63]:


sns.clustermap(pd.crosstab(df['survived'], df['class']), annot = True)

