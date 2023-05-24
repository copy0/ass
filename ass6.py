#!/usr/bin/env python
# coding: utf-8

# # Assignment 06
# ## Data Analytics III
# 1. Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv dataset.
# 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall on the given dataset.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


df = sns.load_dataset('iris')


# ### Basic five questions

# In[4]:


df.head(5)


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# ### Converting species column into numrical value

# In[9]:


df['species'].value_counts()


# In[10]:


df['species'].replace(['setosa', 'versicolor', 'virginica'], [0, 1, 2], inplace = True)


# In[11]:


df.info()


# In[12]:


df.describe()


# ### EDA (Exploratory Data Analysis)

# In[13]:


sns.boxplot(data = df)
plt.show()


# In[14]:


# Defininig function for Outliers Treatement

def Outlier_Treatment(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + (1.5 * IQR)
    lower = Q1 - (1.5 * IQR)
    np.clip(df[col], lower, upper, inplace = True)


# In[15]:


Outlier_Treatment('sepal_width')


# In[16]:


sns.boxplot(data = df)
plt.show()


# In[17]:


sns.pairplot(data = df)


# In[18]:


sns.kdeplot(data = df)


# ### Building Naïve Bayes Classification Model

# #### Setting the value of X and y

# In[19]:


X = df.iloc[ : ,  : -1]
y = df.iloc[ : , -1]


# In[20]:


X


# In[21]:


y


# #### Spliting X and y into traning and testing set

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


y_train.shape


# In[27]:


y_test.shape


# #### Creating and instance of Naïve Bayes classifier

# In[28]:


from sklearn.naive_bayes import GaussianNB


# In[29]:


classify = GaussianNB()


# #### Training the model

# In[30]:


classify.fit(X_train, y_train)


# #### Test the model

# In[31]:


y_predict = classify.predict(X_test)


# In[32]:


y_predict


# #### Evaluate the model

# In[33]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[34]:


confusion_matrix(y_test,  y_predict)


# In[35]:


accuracy_score(y_test,  y_predict)


# In[36]:


precision_score(y_test,  y_predict, average = None)


# In[37]:


recall_score(y_test,  y_predict, average = None)

