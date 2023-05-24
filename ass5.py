#!/usr/bin/env python
# coding: utf-8

# # Assignment 05
# ## Data Analytics II
# 1. Implement logistic regression using Python/R to perform classification on Social_Network_Ads.csv dataset.
# 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall on the given dataset.

# ### Five Basic Operations

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import warnings


# In[2]:


warnings.filterwarnings('ignore')         # It removes the Warning messages form the outputs


# In[3]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[4]:


df.head(10)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# ### Gender Catgories Change

# In[9]:


df['Gender'].value_counts()


# In[10]:


df['Gender'].replace(['Female', 'Male'], [0, 1], inplace = True)


# In[11]:


df.head(10)


# In[12]:


df.info()


# ### Perform EDA (Exploratory Data Analysis)

# In[13]:


import matplotlib.pyplot as plt


# In[14]:


for i in df.columns:
    plt.figure(figsize = (8, 4))
    sns.boxplot(x = i, data = df)
    plt.show()


# In[15]:


# Droping unique valeues
df['User ID'].value_counts()


# In[16]:


df.drop(['User ID'], axis = 1, inplace = True)


# In[17]:


sns.kdeplot(x = 'Age', data = df)
plt.show()


# In[18]:


sns.kdeplot(x = 'EstimatedSalary', data = df)
plt.show()


# In[19]:


df['EstimatedSalary'].skew()     # Normally Distributed


# In[20]:


df['Age'].skew()                 # Normally Distributed


# In[21]:


sns.countplot(x = 'Gender', data = df)


# In[22]:


sns.countplot(x = 'Purchased', data = df)


# ### Build Logistic Regression Model

# In[23]:


X = df.iloc[ : , : -1]


# In[24]:


y = df.iloc[ : , -1]


# In[25]:


X


# In[26]:


y


# ### splitting data into training and testing

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[29]:


X_train.shape


# In[30]:


X_test.shape


# In[31]:


y_train.shape


# In[32]:


y_test.shape


# ### Scaling of the dataset

# In[33]:


from sklearn.preprocessing import StandardScaler


# In[34]:


scaler = StandardScaler()


# In[35]:


X_train = scaler.fit_transform(X_train)


# In[36]:


X_test = scaler.transform(X_test)


# In[37]:


sns.kdeplot(X_train)


# In[38]:


sns.kdeplot(X_test)


# ### Create Logistic Regression Object

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


LogReg = LogisticRegression()


# ### Train the model

# In[41]:


LogReg.fit(X_train, y_train)


# ### Test the model

# In[42]:


y_predict = LogReg.predict(X_test)


# In[43]:


y_predict


# ### Evaluate the model

# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[45]:


confusion_matrix(y_test, y_predict)


# In[46]:


accuracy_score(y_test, y_predict)


# In[47]:


precision_score(y_test, y_predict)


# In[48]:


recall_score(y_test, y_predict)

