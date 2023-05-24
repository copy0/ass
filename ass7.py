#!/usr/bin/env python
# coding: utf-8

# # Assignment 07
# ## Text Analytics
# 1. Extract Sample document and apply following document preprocessing methods: Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
# 2. Create representation of document by calculating Term Frequency and Inverse Document Frequency.

# In[1]:


import numpy as np
import pandas as pd
import nltk as nlt
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


text = "Machine learning (ML) is a field devoted to understanding and building methods that let machines \"learn\" – that is, methods that leverage data to improve computer performance on some set of tasks. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so."


# In[4]:


text


# ### Tokenization

# In[5]:


from nltk import word_tokenize, sent_tokenize


# In[6]:


nlt.download('punkt')


# In[7]:


word_tokenize(text)


# In[8]:


sent_tokenize(text)


# ### Removing Stop-words

# In[9]:


from nltk.corpus import stopwords


# In[10]:


nlt.download('stopwords')


# In[11]:


stop_words = stopwords.words('english')


# In[12]:


print(stop_words)


# In[13]:


len(stop_words)


# In[14]:


token = word_tokenize(text)
cleaned_token = []


# In[15]:


for word in token:
    if word not in stop_words:
        cleaned_token.append(word)


# In[16]:


print("This is the unclean version: ", token)


# In[17]:


print("This is the clean version: ", cleaned_token)


# In[18]:


len(token)


# In[19]:


len(cleaned_token)


# ### Converting clean token text in to lowercase

# In[20]:


words = [token.lower() 
         for token in cleaned_token if token.isalpha()
        ]


# In[21]:


words


# ### Stemming Process

# In[22]:


from nltk.stem import PorterStemmer


# In[23]:


stemmer = PorterStemmer()          # Creating PorterStemmer() object


# In[24]:


port_stemmer_output = [
    stemmer.stem(word) for word in words
]


# In[25]:


print(port_stemmer_output)


# ### Lemmatization Process

# In[26]:


from nltk.stem import WordNetLemmatizer


# In[27]:


nlt.download('wordnet')


# In[28]:


nlt.download('omw-1.4')


# In[29]:


lemmatizer = WordNetLemmatizer()


# In[30]:


lemmatizer_output = [
    lemmatizer.lemmatize(word) for word in words
]


# In[31]:


print(lemmatizer_output)


# ### Part Of Speech (POS) Tagging

# In[32]:


from nltk import pos_tag


# In[33]:


nlt.download('averaged_perceptron_tagger')


# In[34]:


tagged = pos_tag(lemmatizer_output)


# In[35]:


tagged


# ### Create representation of document by calculating Term Frequency and Inverse Document Frequency.

# In[36]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[37]:


docs = [
    "Sachin is considered to be one of the greatest cricket players",
    "Federer is considered one of the greatest tennis players",
    "Nadal is considered one of the greatest tennis players",
    "Virat is the captain of the Indian cricket team"
]


# In[38]:


docs


# In[39]:


vectorizer = TfidfVectorizer(analyzer = "word", norm = None)


# In[40]:


Mat = vectorizer.fit(docs)


# In[41]:


print(Mat.vocabulary_)


# In[42]:


tfidfMat = vectorizer.fit_transform(docs)


# In[43]:


print(tfidfMat)


# In[44]:


features_names = vectorizer.get_feature_names_out()


# In[45]:


print(features_names)


# In[46]:


dense = tfidfMat.todense()


# In[47]:


denselist = dense.tolist()


# In[48]:


df = pd.DataFrame(denselist, columns = features_names)


# In[49]:


df


# In[50]:


features_names = sorted(vectorizer.get_feature_names_out())


# In[51]:


features_names


# In[52]:


doclist = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4']


# In[53]:


skDocslfldfdf = pd.DataFrame(tfidfMat.todense(), index = sorted(doclist), columns = features_names)


# In[54]:


print(skDocslfldfdf)

