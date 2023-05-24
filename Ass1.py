import numpy as np
import pandas as pd

df = pd.read_csv("titanic.csv")

df.head(10)
df.tail(10)
df.describe()
df.info()
df.isnull().sum()


df['Age'].fillna(df['Age'].mean(), inplace = True)
df.isnull().sum()

df.shape

df.drop(['Cabin'], axis = 1, inplace = True)
df.isnull().sum()

df.drop(['PassengerId'], axis = 1, inplace = True)

df.isnull().sum()

df.info()

df.drop(['Name'], axis = 1, inplace = True)

df.info()
df.drop(['Ticket'], axis = 1, inplace = True)

df.info()

columns = df.columns[df.dtypes == 'int64']

for i in columns:
    df[i] = df[i].astype('int32')

df.info()

columns = df.columns[df.dtypes == 'float64']

for i in columns:
    df[i] = df[i].astype('float32')

df.info()

df.dtypes

df['Sex'].value_counts()

df['Sex'].replace(['male', 'female'], [0, 1], inplace = True)

df.dtypes

df['Sex'] = df['Sex'].astype('int32')

df.dtypes

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

enc = pd.DataFrame(ohe.fit_transform(df[['Embarked']]).toarray())
df = df.join(enc)

df.head()