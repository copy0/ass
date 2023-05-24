import pandas as pd
14 import seaborn as sns
15 import matplotlib.pyplot as plt
16 import warnings
17
18
19
20 CHAPTER 4. ASSIGNMENT 04
19 # In[2]:
20
21
22 warnings.filterwarnings('ignore')
23
24
25 # In[3]:
26
27
28 df = pd.read_csv("HousingData.csv")
29
30
31 # ### Boston Dataset Description
32 # 1. CRIM - per capita crime rate by town
33 # 2. ZN - proportion of residential land zoned for lots over
25,000 sq.ft.
34 # 3. INDUS - proportion of non -retail business acres per town
.
35 # 4. CHAS - Charles River dummy variable (1 if tract bounds
river; 0 otherwise)
36 # 5. NOX - nitric oxides concentration (parts per 10 million)
37 # 6. RM - average number of rooms per dwelling
38 # 6. AGE - proportion of owner -occupied units built prior to
1940
39 # 7. DIS - weighted distances to five Boston employment
centres
40 # 8. RAD - index of accessibility to radial highways
41 # 9. TAX - full -value property -tax rate per 10,000 doller
42 # 10. PTRATIO - pupil -teacher ratio by town
43 # 11. B - $1000(Bk - 0.63)^2$ where Bk is the proportion of
blacks by town
44 # 12. LSTAT - \% lower status of the population
45 # 13. MEDV - Median value of owner -occupied homes in 1000's
doller
46
47 # ### Five basic operations on the dataset
48
49 # In[4]:
50
51
52 df.shape
53
54
21
55 # In[5]:
56
57
58 df.head(10)
59
60
61 # In[6]:
62
63
64 df.describe()
65
66
67 # In[7]:
68
69
70 df.info()
71
72
73 # In[8]:
74
75
76 df.isnull().sum()
77
78
79 # ### Treating Null values
80
81 # In[9]:
82
83
84 df.fillna(df.mean(), inplace=True)
85
86
87 # In[10]:
88
89
90 df.isnull().sum()
91
92
93 # In[11]:
94
95
96 df.corr()
97
22 CHAPTER 4. ASSIGNMENT 04
98
99 # In[12]:
100
101
102 plt.figure(figsize = (12, 10))
103 sns.heatmap(df.corr(), annot = True)
104
105
106 # In[13]:
107
108
109 sns.pairplot(df)
110 plt.show()
111
112
113 # In[14]:
114
115
116 sns.histplot(df['RM'])
117 plt.show()
118
119
120 # In[15]:
121
122
123 sns.histplot(df['PTRATIO'])
124 plt.show()
125
126
127 # In[16]:
128
129
130 sns.histplot(df['LSTAT'])
131 plt.show()
132
133
134 # In[17]:
135
136
137 plt.figure(figsize = (12, 10))
138 sns.boxplot(df)
139 plt.show()
140
23
141
142 # In[18]:
143
144
145 # Defininig function for Outliers Treatement
146
147 def Outlier_Treatment(col):
148 Q1 = df[col].quantile (0.25)
149 Q3 = df[col].quantile (0.75)
150 IQR = Q3 - Q1
151 upper = Q3 + (1.5 * IQR)
152 lower = Q1 - (1.5 * IQR)
153 np.clip(df[col], lower , upper , inplace = True)
154
155
156 # In[19]:
157
158
159 outlier_list = ['CRIM', 'ZN', 'CHAS', 'RM', 'DIS', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
160
161 for i in outlier_list:
162 Outlier_Treatment(i)
163
164
165 # In[20]:
166
167
168 plt.figure(figsize = (12, 10))
169 sns.boxplot(df)
170 plt.show()
171
172
173 # In[21]:
174
175
176 X = df.iloc[: , : -1]
177 Y = df.iloc[: , -1]
178
179
180 # In[22]:
181
182
24 CHAPTER 4. ASSIGNMENT 04
183 X # Test Data set
184
185
186 # In[23]:
187
188
189 Y # Train Data set
190
191
192 # ### Second step: spliting dataset into train and test set
193
194 # In[24]:
195
196
197 from sklearn.model_selection import train_test_split
198
199
200 # In[25]:
201
202
203 X_train , X_test , Y_train , Y_test = train_test_split(X, Y,
test_size = 0.20, random_state = 42)
204
205
206 # In[26]:
207
208
209 X_train.shape
210
211
212 # In[27]:
213
214
215 X_test.shape
216
217
218 # In[28]:
219
220
221 Y_train.shape
222
223
224 # In[29]:
25
225
226
227 Y_test.shape
228
229
230 # ### Third step: Create regressor object
231
232 # In[30]:
233
234
235 from sklearn.linear_model import LinearRegression
236
237
238 # In[31]:
239
240
241 lr = LinearRegression()
242
243
244 # ### Forth step: Train the model
245
246 # In[32]:
247
248
249 lr.fit(X_train , Y_train)
250
251
252 # ### Fifth step: Test Model
253
254 # In[33]:
255
256
257 Y_predict = lr.predict(X_test)
258
259
260 # In[34]:
261
262
263 Y_predict
264
265
266 # ### Sixth step: evaluate the model
267
26 CHAPTER 4. ASSIGNMENT 04
268 # In[35]:
269
270
271 from sklearn.metrics import mean_squared_error ,
mean_absolute_error
272
273
274 # In[36]:
275
276
277 mean_squared_error(Y_test , Y_predict)
278
279
280 # In[37]:
281
282
283 mean_absolute_error(Y_test , Y_predict)
284
285
286 # In[38]:
287
288
289 np.sqrt(mean_squared_error(Y_test , Y_predict))
