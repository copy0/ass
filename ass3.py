import numpy as np
18 import pandas as pd
19 import seaborn as sns

25 df = sns.load_dataset('tips')
26
33 df.shape

38
39 df.head()
40
44
45 df.info()
15
46
51 df.isnull().sum()

56
57 df.describe()
58

62
63 df.groupby('day').describe()

69 df.groupby('day')['total_bill'].describe()
70
71
72 # In[10]:
73
74
75 df[df['day'] == 'Sun']['total_bill'].describe()
76
77
83 df = sns.load_dataset('iris')
84
85
16 CHAPTER 3. ASSIGNMENT 03
86 # In[12]:
87
88
89 df.shape
90
91
92 # In[13]:
93
94
95 df.info()
96
97
98 # In[14]:
99
100
101 df.isnull().sum()
102
103
104 # In[15]:
105
106
107 df.head(10)
108
109
110 # In[16]:
111
112
113 df['species'].value_counts()
114
115
116 # In[17]:
117
118
119 df.describe()
120
121
122 # In[18]:
123
124
125 df.groupby('species').describe()
126
127
128 # In[19]:
17
129
130
131 df.groupby('species').min()
132
133
134 # In[20]:
135
136
137 df.groupby('species')['sepal_length'].min()
