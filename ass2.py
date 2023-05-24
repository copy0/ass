import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("StudentsPerformance.csv")

df.head(10)

df.tail(10)

df.describe()

df.info()

df.shape
df.isnull().sum()

df['math score'].fillna(df['math score'].mean(), inplace =True)

df.isnull().sum()

df['reading score'].fillna(df['reading score'].median(),
inplace = True)

df.isnull().sum()

df['writing score'].fillna(df['writing score'].std(), inplace= True)

df.isnull().sum()


df.dtypes

obj_columns_list = df.columns[df.dtypes == 'object']

for i in obj_columns_list:
    print(df[i].value_counts())
    print("Non unique values: ", df[i].nunique(), end = "\n \n")

df['gender'].fillna('female', inplace = True)

df.isnull().sum()

df[df['race/ethnicity'].isnull()]

df['race/ethnicity'].fillna(method = 'bfill', inplace = True)

df.isnull().sum()
df['parental level of education'].fillna(method='pad',inplace=True)

df['lunch'].fillna(method='ffill', inplace=True)

df['test preparation course'].fillna(method='backfill',inplace=True)
157
158 df.isnull().sum()
159
164 sns.boxplot(data = df)
165 plt.show()
166
167
168 # In[43]:
169
170
171 def Outliers_Treatement(dataframe , columnname):
172 Q1 = dataframe[columnname].quantile (0.25)
173 Q3 = dataframe[columnname].quantile (0.75)
174
175 IQR = Q3 - Q1
176 upper = Q3 + (IQR * 1.5)
177 lower = Q1 - (IQR * 1.5)
178
179 np.clip(dataframe[columnname], lower , upper , inplace =True)
180
184
185 outliers_columns = ['math score', 'reading score', 'writing score']
186
187 for i in outliers_columns:
        Outliers_Treatement(df, i)
    sns.boxplot(data = df)
195 plt.show()

201 sns.kdeplot(df['math score'])
206
207 from sklearn.preprocessing import StandardScaler

213 scaler = StandardScaler()

219 scaler.fit(df[['math score']])

225 df['math score scaler'] = scaler.transform(df[['math score']])
